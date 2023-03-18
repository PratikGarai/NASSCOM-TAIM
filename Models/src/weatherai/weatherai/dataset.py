import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from testing.config import config


class Image2ImageDataset(Dataset):
    '''Dataset class for image relate tasks'''
    def __init__(self, input_paths: List[str], output_paths: List[str], mode: str) -> None:
        super(Image2ImageDataset, self).__init__()
        self.input_paths = input_paths
        self.output_paths = output_paths
        
        if self.mode == "train":
            self.transforms = A.Compose([
                A.Resize(height=config.image_size[0], width=config.image_size[1], interpolation=0),
                A.Rotate(limit=90, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(p=0.5),
                A.Normalize(),
                ToTensorV2()
            ], p=1.0)
        elif self.mode == "valid":
            self.transforms = A.Compose([
                A.Resize(height=config.image_size[0], width=config.image_size[1], interpolation=0),
                A.Normalize(),
                ToTensorV2()
            ], p=1.0)
        else:
            raise NotImplementedError(f"The case where mode={self.mode} has not been implemented try from ['train', 'valid']")

    def __len__(self) -> int:
        return len(self.input_paths)
    
    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        input_path = self.input_paths[ix]
        output_path = self.output_paths[ix]
        
        input_img = np.array(Image.open(input_path))
        output_img = np.array(Image.open(output_path))
        
        transformed = self.transforms(image=input_img, mask=output_img)
        
        image = torch.tensor(transformed['image'], dtype=torch.float32, device=config.device)
        mask = torch.tensor(transformed['mask'], dtype=torch.float32, device=config.device)
        mask = mask.permute(2, 0, 1)
        return {
            "image": image / 255.0,
            "mask": mask / 255.0
        }