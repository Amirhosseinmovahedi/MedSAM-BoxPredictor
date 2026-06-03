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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.Dataset import Dataset
import monai
from monai.metrics import DiceMetric
from segment_anything import sam_model_registry
import argparse
import random
from datetime import datetime
import shutil
import glob

IMAGE_SIZE = 1024

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        box_predictor, 
        enlarge_box_percentage=0.0,
        include_point=True
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.box_predictor = box_predictor 
        self.enlarge_box_percentage = enlarge_box_percentage
        self.include_point = include_point

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        for param in self.box_predictor.parameters():
            param.requires_grad = False


    def forward(self, image, point_1024, point_norm): 

        with torch.no_grad():
            image_embedding = self.image_encoder(image)  

            if point_norm.dim() == 3 and point_norm.shape[1] == 1:
                point_norm_for_predictor = point_norm.squeeze(1)
            else:
                point_norm_for_predictor = point_norm

            predicted_box_norm = self.box_predictor(image_embedding, point_norm_for_predictor)

            if self.enlarge_box_percentage > 0:
                current_x_min_1024 = predicted_box_norm[:, 0] * IMAGE_SIZE
                current_y_min_1024 = predicted_box_norm[:, 1] * IMAGE_SIZE
                current_x_max_1024 = predicted_box_norm[:, 2] * IMAGE_SIZE
                current_y_max_1024 = predicted_box_norm[:, 3] * IMAGE_SIZE

                center_x_1024 = (current_x_min_1024 + current_x_max_1024) / 2
                center_y_1024 = (current_y_min_1024 + current_y_max_1024) / 2
                width_1024 = current_x_max_1024 - current_x_min_1024
                height_1024 = current_y_max_1024 - current_y_min_1024

                enlargement_factor = 1 + (self.enlarge_box_percentage / 100.0)
                
                new_width_1024 = width_1024 * enlargement_factor
                new_height_1024 = height_1024 * enlargement_factor

                enlarged_x_min_1024 = center_x_1024 - (new_width_1024 / 2)
                enlarged_y_min_1024 = center_y_1024 - (new_height_1024 / 2)
                enlarged_x_max_1024 = center_x_1024 + (new_width_1024 / 2)
                enlarged_y_max_1024 = center_y_1024 + (new_height_1024 / 2)

                enlarged_x_min_1024 = torch.clamp(enlarged_x_min_1024, 0.0, IMAGE_SIZE)
                enlarged_y_min_1024 = torch.clamp(enlarged_y_min_1024, 0.0, IMAGE_SIZE)
                enlarged_x_max_1024 = torch.clamp(enlarged_x_max_1024, 0.0, IMAGE_SIZE)
                enlarged_y_max_1024 = torch.clamp(enlarged_y_max_1024, 0.0, IMAGE_SIZE)

                min_size = 1.0 
                enlarged_x_max_1024 = torch.max(enlarged_x_max_1024, enlarged_x_min_1024 + min_size)
                enlarged_y_max_1024 = torch.max(enlarged_y_max_1024, enlarged_y_min_1024 + min_size)


                predicted_box_1024 = torch.stack([enlarged_x_min_1024, enlarged_y_min_1024, enlarged_x_max_1024, enlarged_y_max_1024], dim=1)
            else:
                predicted_box_1024 = predicted_box_norm * IMAGE_SIZE

            if point_1024.dim() == 2:
                point_1024_for_encoder = point_1024[:, None, :] 
            else:
                point_1024_for_encoder = point_1024 

            point_labels = torch.ones(point_1024_for_encoder.shape[0], point_1024_for_encoder.shape[1], device=image.device)

        if self.include_point == True:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_1024_for_encoder, point_labels), 
                boxes=predicted_box_1024, 
                masks=None,
            )
        else:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, 
                boxes=predicted_box_1024, 
                masks=None,
            )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  
            image_pe=self.prompt_encoder.get_dense_pe(),  
            sparse_prompt_embeddings=sparse_embeddings,  
            dense_prompt_embeddings=dense_embeddings, 
            multimask_output=False,
        )
        
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks