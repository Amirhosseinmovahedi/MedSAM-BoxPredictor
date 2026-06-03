import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
join = os.path.join
import torch
from torch.utils.data import Dataset
import random
import glob
import torch

from PIL import Image
import cv2

IMAGE_SIZE = 1024

class BoxPredictorDataset(Dataset):
    def __init__(self, data_root, img_subdir="imgs", gt_subdir="gts", perturbation=40):
        self.data_root = data_root
        self.gt_path = join(data_root, gt_subdir)
        self.img_path = join(data_root, img_subdir)
        self.perturbation = perturbation

        if not os.path.exists(self.gt_path):
            raise FileNotFoundError(f"Ground truth path not found: {self.gt_path}")
        if not os.path.exists(self.img_path):
            raise FileNotFoundError(f"Image path not found: {self.img_path}")

        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.png"), recursive=True)
        )

        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        print(f"Number of files in {img_subdir}/{gt_subdir}: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):

        img_name = os.path.basename(self.gt_path_files[index])

        img_512 = Image.open(join(self.img_path, img_name)).convert('RGB')
        img_1024 = img_512.resize((1024, 1024), Image.BILINEAR)  
        img_1024 = np.array(img_1024) / 255.0  
        img_1024 = np.transpose(img_1024, (2, 0, 1)) 

        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        
        gt_512 = Image.open(self.gt_path_files[index]).convert('L')  
        gt_1024 = gt_512.resize((1024, 1024), Image.NEAREST)  
        gt = np.array(gt_1024)
        

        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + img_name
        )
        

        label_ids = np.unique(gt)[1:]
        
        H, W = gt.shape 

        original_H, original_W = IMAGE_SIZE, IMAGE_SIZE 

        if len(label_ids) == 0: 

            point_x_norm = 0.5 
            point_y_norm = 0.5 
            point_coords = np.array([[point_x_norm, point_y_norm]], dtype=np.float32)

            half_side_norm = 10.0 / IMAGE_SIZE 
            gt_bbox = np.array([
                point_x_norm - half_side_norm, point_y_norm - half_side_norm,
                point_x_norm + half_side_norm, point_y_norm + half_side_norm
            ], dtype=np.float32)
            gt2D = np.zeros_like(gt, dtype=np.uint8) 
        else:
            gt2D = np.uint8(
                gt == random.choice(label_ids.tolist())
            ) 
            assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"


            y_indices, x_indices = np.where(gt2D > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                center_x_gt_mask_256 = np.mean(x_indices)
                center_y_gt_mask_256 = np.mean(y_indices)

                scale_factor_gt_to_1024 = IMAGE_SIZE / gt.shape[0]
                center_x_gt_mask_1024 = center_x_gt_mask_256 * scale_factor_gt_to_1024
                center_y_gt_mask_1024 = center_y_gt_mask_256 * scale_factor_gt_to_1024


                perturbation = self.perturbation
                point_x_1024 = center_x_gt_mask_1024 + random.uniform(-1 * perturbation, perturbation) 
                point_y_1024 = center_y_gt_mask_1024 + random.uniform(-1 * perturbation, perturbation)
                
 
                point_x_1024 = np.clip(point_x_1024, 0, original_W - 1)
                point_y_1024 = np.clip(point_y_1024, 0, original_H - 1)
  
                x_idx = int(round(point_x_1024))
                y_idx = int(round(point_y_1024))

                if gt2D[y_idx, x_idx] != 1:
                    mask_indices = np.argwhere(gt2D == 1)
                    random_index = np.random.choice(len(mask_indices))
                    random_point = mask_indices[random_index]
                    point_y_1024, point_x_1024 = random_point



                point_x_norm = point_x_1024 / original_W
                point_y_norm = point_y_1024 / original_H
                point_coords = np.array([[point_x_norm, point_y_norm]], dtype=np.float32) 


                x_min_gt_mask_256, y_min_gt_mask_256 = np.min(x_indices), np.min(y_indices)
                x_max_gt_mask_256, y_max_gt_mask_256 = np.max(x_indices), np.max(y_indices)

                x_min_gt_mask_1024 = x_min_gt_mask_256 * scale_factor_gt_to_1024
                y_min_gt_mask_1024 = y_min_gt_mask_256 * scale_factor_gt_to_1024
                x_max_gt_mask_1024 = x_max_gt_mask_256 * scale_factor_gt_to_1024
                y_max_gt_mask_1024 = y_max_gt_mask_256 * scale_factor_gt_to_1024
                

                gt_half_width_1024 = (x_max_gt_mask_1024 - x_min_gt_mask_1024) / 2.0
                gt_half_height_1024 = (y_max_gt_mask_1024 - y_min_gt_mask_1024) / 2.0
                

                gt_bbox_1024 = np.array([
                    x_min_gt_mask_1024,
                    y_min_gt_mask_1024,
                    x_max_gt_mask_1024,
                    y_max_gt_mask_1024
                ], dtype=np.float32)

                gt_bbox_1024[0] = np.clip(gt_bbox_1024[0], 0.0, original_W - 1.0)
                gt_bbox_1024[1] = np.clip(gt_bbox_1024[1], 0.0, original_H - 1.0)
                gt_bbox_1024[2] = np.clip(gt_bbox_1024[2], 0.0, original_W - 1.0)
                gt_bbox_1024[3] = np.clip(gt_bbox_1024[3], 0.0, original_H - 1.0)

                gt_bbox_1024[2] = np.maximum(gt_bbox_1024[2], gt_bbox_1024[0] + 1.0)
                gt_bbox_1024[3] = np.maximum(gt_bbox_1024[3], gt_bbox_1024[1] + 1.0)

                gt_bbox = gt_bbox_1024 / IMAGE_SIZE

            else:
                H, W = gt2D.shape 
                point_x_norm = 0.5
                point_y_norm = 0.5
                point_coords = np.array([[point_x_norm, point_y_norm]], dtype=np.float32)
                
                half_side_norm = 10.0 / IMAGE_SIZE
                gt_bbox = np.array([
                    point_x_norm - half_side_norm, point_y_norm - half_side_norm,
                    point_x_norm + half_side_norm, point_y_norm + half_side_norm
                ], dtype=np.float32)

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(point_coords).float(), 
            torch.tensor(gt_bbox).float(),      
            img_name,
        )