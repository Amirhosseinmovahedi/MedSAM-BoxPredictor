import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
from segment_anything import sam_model_registry

from utils.builders import build_box_predictor, build_medsam, build_dataset

join = os.path.join

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

IMAGE_SIZE = 1024

def validate_model(model, val_dataloader, seg_loss, ce_loss, device, box_available):
    model.eval()  
    val_loss = 0
    iou_list = []
    dice_list = []
    recall_list = []
    precision_list = []

    with torch.no_grad():
        for image, gt2D, points_1024, points_norm, _ in tqdm(val_dataloader, desc="Validation"):
            image, gt2D = image.to(device), gt2D.to(device).float()
            points_1024 = points_1024.to(device)
            if box_available:
                points_norm = points_norm.to(device)
                pred = model(image, points_1024, points_norm) 
            else:
                pred = model(image, points_1024)
            
            loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
            val_loss += loss.item()

            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            batch_size = image.size(0)
            for i in range(batch_size):
                pred_single = pred_binary[i].view(-1)
                gt_single = gt2D[i].view(-1)
                tp = (pred_single * gt_single).sum().item()
                fp = (pred_single * (1 - gt_single)).sum().item()
                fn = ((1 - pred_single) * gt_single).sum().item()
                epsilon = 1e-6
                iou = tp / (tp + fp + fn + epsilon)
                iou_list.append(iou)
                dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
                dice_list.append(dice)
                recall = tp / (tp + fn + epsilon)
                recall_list.append(recall)
                precision = tp / (tp + fp + epsilon)
                precision_list.append(precision)

    avg_iou = np.mean(iou_list) if len(iou_list) > 0 else 0.0
    avg_dice = np.mean(dice_list) if len(dice_list) > 0 else 0.0
    avg_recall = np.mean(recall_list) if len(recall_list) > 0 else 0.0
    avg_precision = np.mean(precision_list) if len(precision_list) > 0 else 0.0
    
    avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')

    model.train()

    return avg_val_loss, avg_dice, avg_iou, avg_recall, avg_precision, {
        'iou_list': iou_list,
        'dice_list': dice_list,
        'recall_list': recall_list,
        'precision_list': precision_list
    }


def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-val_data_path",
        type=str,
        default="data/BUSI",
        help="Path to validation PNG files; expects subfolders like imgs_brain_test_png and gts_brain_test_png."
    )
    parser.add_argument(
        "-model_type", type=str, default="vit_b",
        help="Type of SAM model (e.g., vit_b, vit_l, vit_h)."
    )
    parser.add_argument(
        "-sam_checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to the original MedSAM checkpoint (e.g., medsam_vit_b.pth)."
    )
    parser.add_argument(
        "-medsam_trained_model", type=str, 
        default=None,
        help="Path to your *trained* MedSAM model checkpoint (e.g., medsam_model_best.pth)."
    )
    parser.add_argument(
        "-box_predictor_checkpoint", type=str, 
        default=None,
        help="Path to the trained BoxPredictor checkpoint"
    )
    parser.add_argument(
        "-enlarge_box_percentage", type=float, default=20.0,
        help="Percentage to enlarge the predicted bounding box (e.g., 20 for 20%)"
    )
    parser.add_argument(
        "-device", type=str, default="cuda:0",
        help="Device to use for computation (e.g., cuda:0 or cpu)."
    )
    parser.add_argument(
        "-point_perturbation", type=int, default=91,
        help="Maximum perturbation in pixels for the input point."
    )
    parser.add_argument(
        "-batch_size", type=int, default=1,
        help="Batch size for validation DataLoader (recommended 1 to prevent OOM)."
    )
    parser.add_argument(
        "-num_workers", type=int, default=2,
        help="Number of workers for DataLoader."
    )
    parser.add_argument(
        "--use_box_predictor",
        action="store_true",
        default=False,
        help="If set, use BoxPredictor to generate box prompt (default: False)"
    )

    parser.add_argument(
        "--use_point",
        action="store_true",
        default=False,
        help="If set, use point when we are using boxPredictor"
    )

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        print("CUDA is not available. Please ensure you have a CUDA-compatible GPU and PyTorch is installed correctly with CUDA support.")
        sys.exit(1)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")


    print(f"Loading MedSAM model type: {args.model_type} from {args.sam_checkpoint}")
    if not os.path.exists(args.sam_checkpoint):
        print(f"Error: Original MedSAM checkpoint not found at {args.sam_checkpoint}")
        sys.exit(1)

    box_available = False
    if args.box_predictor_checkpoint != None:
        print(f"Loading trained BoxPredictor from {args.box_predictor_checkpoint}")
        box_predictor = build_box_predictor().to(device)

        if os.path.exists(args.box_predictor_checkpoint):
            box_predictor_checkpoint = torch.load(args.box_predictor_checkpoint, map_location="cpu")
            box_predictor.load_state_dict(box_predictor_checkpoint["model"])
            print("BoxPredictor weights loaded successfully.")
            box_available = True
        else:
            print(f"Error: BoxPredictor checkpoint not found at {args.box_predictor_checkpoint}!")
            sys.exit(1)
            
    sam_model_base = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)

    if args.use_box_predictor == False:
        medsam_model = build_medsam(2,
            image_encoder=sam_model_base.image_encoder,
            mask_decoder=sam_model_base.mask_decoder,
            prompt_encoder=sam_model_base.prompt_encoder,
        ).to(device)
    elif args.use_box_predictor == True:
        medsam_model = build_medsam(1,
            image_encoder=sam_model_base.image_encoder,
            mask_decoder=sam_model_base.mask_decoder,
            prompt_encoder=sam_model_base.prompt_encoder,
            box_predictor=box_predictor,
            enlarge_box_percentage=args.enlarge_box_percentage,
            use_point=args.use_point
        ).to(device)

    if args.medsam_trained_model != None:
        print(f"Loading trained MedSAM model from {args.medsam_trained_model}")
        if os.path.exists(args.medsam_trained_model):
            checkpoint = torch.load(args.medsam_trained_model, map_location=device)
            medsam_model.load_state_dict(checkpoint["model"])
            print("Trained MedSAM model loaded successfully.")
        else:
            print(f"Error: Trained MedSAM model checkpoint not found at {args.medsam_trained_model}!")
            sys.exit(1)

    val_dataset = build_dataset(args.val_data_path, point_perturbation=args.point_perturbation)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Number of validation samples: {len(val_dataset)}")

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    print("\nStarting evaluation...")
    avg_val_loss, avg_val_dice, avg_val_iou, avg_val_recall, avg_val_precision, metric_lists = validate_model(
        medsam_model,
        val_dataloader,
        seg_loss,
        ce_loss,
        device,
        box_available
    )

    print(f"\n--- Evaluation Results (Per-Image Averaging) ---")
    print(f"Average Validation Loss: {avg_val_loss:4f}")
    print(f"        IoU (Jaccard): {avg_val_iou:.4f}")
    print(f"        F1 (Dice): {avg_val_dice:.4f}")
    print(f"             Recall: {avg_val_recall:.4f}")
    print(f"          Precision: {avg_val_precision:.4f}")
    print(f"    Number of Images: {len(metric_lists['iou_list'])}")
    print(f"-------------------------------------------------")
    
    print(f"\n--- Standard Deviations ---")
    print(f"    IoU Std: {np.std(metric_lists['iou_list']):.4f}")
    print(f"   Dice Std: {np.std(metric_lists['dice_list']):.4f}")
    print(f" Recall Std: {np.std(metric_lists['recall_list']):.4f}")
    print(f"   Prec Std: {np.std(metric_lists['precision_list']):.4f}")
    print(f"---------------------------")
    
    print(f"\nVerification: F1/Dice ({avg_val_dice:.4f}) >= IoU ({avg_val_iou:.4f}): {avg_val_dice >= avg_val_iou}")
    
    results = {
        "loss": avg_val_loss,
        "dice_f1": avg_val_dice,
        "iou": avg_val_iou,
        "recall": avg_val_recall,
        "precision": avg_val_precision,
        "metric_lists": metric_lists
    }
    return results


if __name__ == "__main__":
    run_evaluation()