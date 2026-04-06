import torch
import os
import cv2
import numpy as np
import math
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# def preprocess_image_rescaleT(input_source, size=320):
#     """
#     读取、Resize图像, 并将其转换为张量。 
#     """
#     valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
#     if isinstance(input_source, str):
#         if os.path.isdir(input_source):
#             image_paths = [os.path.join(input_source, f) for f in os.listdir(input_source) if f.lower().endswith(valid_extensions)]
#         elif os.path.isfile(input_source):
#             image_paths = [input_source]
#     else: 
#         image_paths = input_source
    
#     if not image_paths:
#         print(f"错误: 无法识别的输入源 {input_source}")
#         return None
    
#     processed_tensors = []
    
#     for path in tqdm(image_paths):
#         # img = cv2.imread(path)
#         img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
#         if img is None:
#             print(f"无法读取图片 {path}")
#             continue
        
#         # # 原图尺寸（H，W)
#         # h, w = img.shape[:2]
        
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2默认以BGR格式读取图像, 转换为RGB
#         img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)  # 强制缩放到指定大小
        
#         # 归一到[0, 1]并max缩放
#         img = img.astype(np.float32) / 255.0  
#         # img_max = np.max(img)
#         # if img_max > 0:
#         #     img /= img_max  
            
#         # 标准化
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#         img = (img - mean) / std 
        
#         img = img.transpose(2, 0, 1)  # HWC -> CHW
#         processed_tensors.append(torch.from_numpy(img))
        
#     if len(processed_tensors) == 0:
#         return None
    
#     # 将所有 Tensor 堆叠成一个 Batch: (N, 3, 320, 320)
#     batch_tensor = torch.stack(processed_tensors, dim=0)
    
#     return batch_tensor


class RescaleTAndNormalize(object):
    """
    适配官方 Dataset 的转换操作
    """
    def __init__(self, output_size=320):
        self.output_size = output_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # 1. Resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2默认以BGR格式读取图像, 转换为RGB
        image = cv2.resize(image, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)

        # 2. 归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # 3. 标准化
        image = (image - self.mean) / self.std

        # 4. HWC -> CHW
        image = image.transpose((2, 0, 1))
        
        # 标签也增加一个通道维度并转为 CHW
        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]
        label = label.transpose((2, 0, 1))

        return {
            'imidx': imidx,
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(label).float()
        }           
 
    
class RescaleAndNormalize(object):
    """
    适配官方 Dataset 的转换操作，保持长宽比缩放, 并根据distribution填充至 output_size
    """
    def __init__(self, output_size=320, distribution=(0.5, 0.5)):
        """
        :param distribution: (h_ratio, w_ratio) 范围在 [0, 1]之间, 表示在高度和宽度方向上如何分配填充像素,e.g.:
                            (0.5, 0.5) 中心填充
                            (0, 0) 左上对齐, 向右下填充
                            (1, 1) 右下对其, 向左上填充
                            (0, 0.5) 顶部居中对齐, 上边不补, 下边补齐高度差, 左右平分宽度差。
        """
        self.output_size = output_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.distribution = distribution
    
    
    def get_target_size(self, orig_w, orig_h):
        """
        基于原图尺寸和缩放比例计算目标宽高, 缩放到最长边等于 output_size, 短边按比例缩放
        """
        scale_ratio = min(self.output_size / orig_h, self.output_size / orig_w)
        target_w = int(orig_w * scale_ratio)
        target_h = int(orig_h * scale_ratio)
        return target_w, target_h, scale_ratio


    def calculate_padding(self, current_w, current_h):
        """
        计算为了达到目标尺寸，上下左右各需要补多少像素
        """
        delta_w = max(0, self.output_size - current_w)
        delta_h = max(0, self.output_size - current_h)
        
        # 按照比例分配填充像素
        pad_top = int(delta_h * self.distribution[0])
        pad_bottom = delta_h - pad_top
        
        pad_left = int(delta_w * self.distribution[1])
        pad_right = delta_w - pad_left
        
        return {
            "top": pad_top,
            "bottom": pad_bottom,
            "left": pad_left,
            "right": pad_right
        }
        

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2默认以BGR格式读取图像, 转换为RGB
        
        # 1. 计算保持长宽比的缩放比例
        h, w = image.shape[:2]
        new_h, new_w, scale = self.get_target_size(w, h)
        
        # 2. Resize
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 3. 填充
        pads = self.calculate_padding(new_w, new_h)
        
        image = np.pad(image, (
            (pads["top"], pads["bottom"]), 
            (pads["left"], pads["right"]), 
            (0, 0)
        ), mode='constant', constant_values=0)
        
        if len(label.shape) == 2:
            label = np.pad(label, (
                (pads["top"], pads["bottom"]), 
                (pads["left"], pads["right"])
            ), mode='constant', constant_values=0)
        else:
            label = np.pad(label, (
                (pads["top"], pads["bottom"]), 
                (pads["left"], pads["right"]), 
                (0, 0)
            ), mode='constant', constant_values=0)
        
        # 4. 归一化与标准化
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        # 5. HWC -> CHW
        image = image.transpose((2, 0, 1))
        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]
        label = label.transpose((2, 0, 1))

        return {
            'imidx': imidx,
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(label).float(),
            'pad_info': pads,
            'scale': scale
        }           
       

