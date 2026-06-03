import numpy as np
import os
from PIL import Image
import glob
join = os.path.join
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.functions import resize_image_and_mask, get_center_point_with_perturbation

IMAGE_SIZE = 1024

class Dataset(Dataset):
    def __init__(self, data_root, point_perturbation=40, is_validation=True):
        self.data_root = data_root
        if is_validation:
            self.gt_path = join(data_root, "gts_val")
            self.img_path = join(data_root, "imgs_val")
        else:
            self.gt_path = join(data_root, "gts")
            self.img_path = join(data_root, "imgs")
        self.point_perturbation = point_perturbation
        
        self.img_files = sorted(glob.glob(join(self.img_path, "*.png")))
        
        self.img_files = [
            file for file in self.img_files
            if os.path.isfile(join(self.gt_path, os.path.basename(file)))
        ]
        
        print(f"Dataset initialized with {len(self.img_files)} images from {self.data_root}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        img_name = os.path.basename(img_path)
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image) / 255.0
        
        mask_path = join(self.gt_path, img_name)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        
        image_1024, mask_1024 = resize_image_and_mask(image, mask, target_size=1024)
        image_1024 = np.transpose(image_1024, (2, 0, 1))
        
        point_1024 = get_center_point_with_perturbation(mask_1024, self.point_perturbation)
        
        if point_1024 is None:
            point_1024 = np.array([512.0, 512.0])

        point_norm = point_1024 / IMAGE_SIZE
        
        return (
                torch.tensor(image_1024).float(),
                torch.tensor(mask_1024[None, :, :]).long(),
                torch.tensor(point_1024).float(),
                torch.tensor(point_norm).float(),
                img_name,
        )