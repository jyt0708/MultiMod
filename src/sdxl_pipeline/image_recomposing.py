import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
from PIL import Image
from torch.utils.data import DataLoader
from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
from u2net.model.u2net import U2NET


class SDXLControlNetInpainter:
    def __init__(self, u2net_model, controlnet_id, base_model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        # CPU 不支持 float16，必须用 float32
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.u2net = u2net_model.to(self.device)

        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id, 
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model_id,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            variant="fp16" if self.device == "cuda" else None,
            use_safetensors=True
        )
        
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload() 
        else:
            self.pipe.to(self.device)
            
        self.pipe.vae.enable_slicing()

    def get_canny_map(self, image_pil, k_size=19, sigma=0.01):
        image_np = np.array(image_pil)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) 
        median = np.median(gray)
        
        low_threshold = int(max(0, (1.0 - sigma) * median))
        high_threshold = int(min(255, (1.0 + sigma) * median))
        
        blurred_for_canny = cv2.GaussianBlur(gray, (k_size, k_size), 0)
        edges = cv2.Canny(blurred_for_canny, low_threshold, high_threshold)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)
    
    def post_process_mask(self, mask_np, dilation_iteration=20, blur_kernel=9):
        """
        对掩码进行后处理：扩张 + 高斯模糊
        """
        kernel = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=dilation_iteration)
        
        if blur_kernel % 2 == 0: blur_kernel += 1
        blurred_mask = cv2.GaussianBlur(dilated_mask, (blur_kernel, blur_kernel), 0)
        return blurred_mask

    def get_mask_from_u2net(self, image_path, threshold=128):
        from src.data_preparation.inference import inference, RescaleAndNormalize
        from u2net.data_loader import SalObjDataset
        
        salobj_dataset = SalObjDataset(
            img_name_list=[image_path], 
            lbl_name_list=[], 
            transform=RescaleAndNormalize(output_size=320)
        )
        salobj_dataloader = DataLoader(salobj_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        heatmap = inference(self.u2net, salobj_dataloader, self.device)
        _, binary_mask = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
        processed_mask = self.post_process_mask(binary_mask)
        
        return Image.fromarray(processed_mask).convert("L")
    
    def prepare_assets(self, image_path, target_size=1024):
        """
        一次性生成所有推理所需的素材
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图片: {image_path}")
           
        init_image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = init_image.size
 
        # 计算缩放并对齐 64
        scale = min(target_size / max(orig_w, orig_h), 1.0)
        new_w = int((orig_w * scale) // 64) * 64
        new_h = int((orig_h * scale) // 64) * 64
 
        init_image = init_image.resize((new_w, new_h), Image.LANCZOS)
       
        print(f"正在预处理: {new_w}x{new_h}")
        canny_image = self.get_canny_map(init_image)
        mask_image = self.get_mask_from_u2net(image_path)
        mask_image = mask_image.resize((new_w, new_h), Image.BILINEAR)
       
        return init_image, canny_image, mask_image, new_w, new_h

    def generate(self, prompt, negative_prompt, assets, **kwargs):
        """
        推理逻辑
        """
        init_image, canny_image, mask_image, w, h = assets
       
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            control_image=canny_image,
            num_inference_steps=kwargs.get("steps", 30),
            controlnet_conditioning_scale=kwargs.get("control_scale", 0.8),  # 值越高，生成的图像就越严格地遵守controlnet的引导
            strength=kwargs.get("strength", 0.99),
            guidance_scale=kwargs.get("cfg", 9.0),
            controlnet_guidance_end=0.4,
            width=w,
            height=h,
        )
        return output.images[0]
 

if __name__ == "__main__":
    def load_model(weight=r"C:\Users\jytna\OneDrive - TUM\MM\MultiMod\u2net.pth"):
        net = U2NET(3, 1)
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(weight))
            net.cuda()
        else:
            net.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
        net.eval()
        return net
   
    net = load_model()
    inpainter = SDXLControlNetInpainter(
        u2net_model=net,    
        controlnet_id="diffusers/controlnet-canny-sdxl-1.0",
        base_model_id="stabilityai/stable-diffusion-xl-base-1.0" # "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    )
 
    # 配置图片路径及其对应的主体描述
    # 格式：{ r"图片路径": "主体描述" }
    tasks = {
        r"C:\Users\jytna\OneDrive - TUM\MM\dackel.jpg": "A tiger",
        r"C:\Users\jytna\OneDrive - TUM\MM\dackel.jpg":  "A black dachshund dog",
        # r"C:\Users\jytna\OneDrive - TUM\MM\pen.jpg":    "A futuristic fountain pen",
        # r"C:\Users\jytna\OneDrive - TUM\MM\fish.jpg":   "A shark",
        # r"C:\Users\jytna\OneDrive - TUM\MM\glass.jpg":  "A crystal wine glass with blue liquid",
        # r"C:\Users\jytna\OneDrive - TUM\MM\flower.jpg": "A red rose",
    }
   
 
    style_suffix = ", clean lines"
    n_prompt = "low quality, blurry, distorted"
 
    test_scales = [0.6]
    output_dir = "image_results"
    os.makedirs(output_dir, exist_ok=True)
 
    # 开始迭代
    for img_path, subject in tasks.items():            
        base_name = os.path.splitext(os.path.basename(img_path))[0]
 
        # 预处理
        try:
            assets = inpainter.prepare_assets(img_path, target_size=1024)
            init_img, canny_img, mask_img, _, _ = assets
           
            # 保存 Debug 图片
            # canny_img.save(os.path.join(output_dir, f"{base_name}_z_canny.png"))
            # mask_img.save(os.path.join(output_dir, f"{base_name}_z_mask.png"))
            print(f"--- 预处理完成: {base_name} ---")
        except Exception as e:
            print(f"预处理失败 {img_path}: {e}")
            continue
       
 
        # 动态合成当前图片的 Prompt
        current_prompt = f"{subject}{style_suffix}"
       
        print(f"\n[{base_name}] Prompt: {current_prompt}")
        print("-" * 30)
 
        intermediates_saved = False
 
        for scale in test_scales:
            print(f"Inference: {base_name} | Scale: {scale}")
           
            result = inpainter.generate(
                prompt=current_prompt,
                negative_prompt=n_prompt,
                assets=assets,
                control_scale=scale
            )
               
            save_path = os.path.join(output_dir, f"{base_name}_scale_{scale}.png")
            result.save(save_path)
 
    print("\n任务全部完成！")
 