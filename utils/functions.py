import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
join = os.path.join
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torch

from PIL import Image
import cv2

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_point(point, ax, marker_size=10):
    ax.plot(point[0], point[1], 'ro', markersize=marker_size)

def show_box(box, ax, edgecolor='cyan', linewidth=2):
    x_min, y_min, x_max, y_max = box
    box_width = x_max - x_min
    box_height = y_max - y_min
    rect = plt.Rectangle((x_min, y_min), box_width, box_height, edgecolor=edgecolor, facecolor='none', lw=linewidth)
    ax.add_patch(rect)

def get_center_point_with_perturbation(mask, perturbation=40):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None

    center_y = int(np.mean(y_indices))
    center_x = int(np.mean(x_indices))

    perturb_x = random.randint(-perturbation, perturbation)
    perturb_y = random.randint(-perturbation, perturbation)
    
    perturbed_x = center_x + perturb_x
    perturbed_y = center_y + perturb_y

    H, W = mask.shape
    perturbed_x = max(0, min(W-1, perturbed_x))
    perturbed_y = max(0, min(H-1, perturbed_y))

    if mask[perturbed_y, perturbed_x] > 0:
        return np.array([perturbed_x, perturbed_y])
    else:
        valid_indices = np.where(mask > 0)
        if len(valid_indices[0]) > 0:
            random_idx = random.randint(0, len(valid_indices[0]) - 1)
            return np.array([valid_indices[1][random_idx], valid_indices[0][random_idx]])
        else:
            return None

def resize_image_and_mask(image, mask, target_size=1024):

    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    image_resized = image_pil.resize((target_size, target_size), Image.BILINEAR)
    image_resized = np.array(image_resized) / 255.0

    mask_resized = cv2.resize(mask.astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    return image_resized, mask_resized


def calculate_iou(boxes1, boxes2):
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()

    x_min_inter = torch.max(boxes1[:, 0], boxes2[:, 0])
    y_min_inter = torch.max(boxes1[:, 1], boxes2[:, 1])
    x_max_inter = torch.min(boxes1[:, 2], boxes2[:, 2])
    y_max_inter = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_width = (x_max_inter - x_min_inter).clamp(0)
    inter_height = (y_max_inter - y_min_inter).clamp(0)
    intersection = inter_width * inter_height
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection

    iou = intersection / (union + 1e-6)
    return iou, union 

def generalized_iou_loss(boxes1, boxes2, reduction='mean'):
    iou, union = calculate_iou(boxes1, boxes2) 

    x_min_C = torch.min(boxes1[:, 0], boxes2[:, 0])
    y_min_C = torch.min(boxes1[:, 1], boxes2[:, 1])
    x_max_C = torch.max(boxes1[:, 2], boxes2[:, 2])
    y_max_C = torch.max(boxes1[:, 3], boxes2[:, 3])

    C_width = (x_max_C - x_min_C).clamp(0)
    C_height = (y_max_C - y_min_C).clamp(0)
    area_C = C_width * C_height

    giou = iou - (area_C - union) / (area_C + 1e-6) 

    loss = 1.0 - giou

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else: # 'none'
        return loss
    
def get_center_point_with_perturbation(mask, perturbation=40):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None

    center_y = int(np.mean(y_indices))
    center_x = int(np.mean(x_indices))

    perturb_x = random.randint(-perturbation, perturbation)
    perturb_y = random.randint(-perturbation, perturbation)
    
    perturbed_x = center_x + perturb_x
    perturbed_y = center_y + perturb_y

    H, W = mask.shape
    perturbed_x = max(0, min(W-1, perturbed_x))
    perturbed_y = max(0, min(H-1, perturbed_y))

    if mask[perturbed_y, perturbed_x] > 0:
        return np.array([perturbed_x, perturbed_y])
    else:
        valid_indices = np.where(mask > 0)
        if len(valid_indices[0]) > 0:
            random_idx = random.randint(0, len(valid_indices[0]) - 1)
            return np.array([valid_indices[1][random_idx], valid_indices[0][random_idx]])
        else:
            return None
        

def resize_image_and_mask(image, mask, target_size=1024):
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    image_resized = image_pil.resize((target_size, target_size), Image.BILINEAR)
    image_resized = np.array(image_resized) / 255.0

    mask_resized = cv2.resize(mask.astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    return image_resized, mask_resized