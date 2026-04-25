from fastapi import FastAPI
from pydantic import BaseModel
import torch
from sdxl_pipeline.smart_composition import SDXLControlNetInpainter
import uvicorn
import os
from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
from u2net.model.u2net import U2NET

app = FastAPI()

# 加载 u2net 和 Inpainter
def load_detector(weight=r"D:\MM\MultiMod\u2net.pth"):
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(weight, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    net.eval()
    if torch.cuda.is_available(): net.cuda()
    return net

net = load_detector()
inpainter = SDXLControlNetInpainter(
    u2net_model=net,    
    controlnet_id="diffusers/controlnet-canny-sdxl-1.0",
    base_model_id="stabilityai/stable-diffusion-xl-base-1.0" 
)

class SDXLRequest(BaseModel):
    image_path: str
    prompt: str
    negative_prompt: str
    output_path: str

@app.post("/inpaint")
async def do_inpaint(req: SDXLRequest):
    try:
        assets = inpainter.prepare_assets(req.image_path, target_size=1024)
        result_img = inpainter.generate(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            assets=assets,
            steps=30,
            control_scale=0.3
        )
        result_img.save(req.output_path)
        return {"status": "success", "saved_at": req.output_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)