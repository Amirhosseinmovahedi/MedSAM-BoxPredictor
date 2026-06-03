import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from monai.metrics import DiceMetric
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from utils.functions import get_center_point_with_perturbation, resize_image_and_mask


class Dataset(Dataset):
    def __init__(self, data_root, point_perturbation=40, is_validation=False):
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
        
        print(f"Number of images: {len(self.img_files)}")

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
        

        point = get_center_point_with_perturbation(mask_1024, self.point_perturbation)
        
        if point is None:

            point = np.array([512, 512])
        
        return (
            torch.tensor(image_1024).float(),
            torch.tensor(mask_1024[None, :, :]).long(),
            torch.tensor(point).float(),
            img_name,
        )
