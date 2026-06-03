import os
import argparse
import shutil
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

import monai
from monai.metrics import DiceMetric

from segment_anything import sam_model_registry

join = os.path.join
from datasets.Dataset import Dataset
from utils.functions import show_mask, show_point
from models.medsam.train.medsam_with_box import MedSAM
from models.box_predictor.box_predictor import BoxPredictor

torch.manual_seed(2023)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

IMAGE_SIZE = 1024.0

def sanity_check_dataset(dataset_path, point_perturbation=40, is_validation=False):
    """Sanity check for the dataset"""
    test_dataset = Dataset(dataset_path, point_perturbation, is_validation)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    
    for step, (image, gt, points_1024, points_norm, names_temp) in enumerate(test_dataloader):
        print(f"Dataset: {'Validation' if is_validation else 'Training'}")
        print(f"Image shape: {image.shape}, GT shape: {gt.shape}, Points_1024 shape: {points_1024.shape}, Points_norm shape: {points_norm.shape}")
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs = axs.flatten()
        
        for i in range(min(4, len(image))):
            idx = i
            axs[i].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
            show_mask(gt[idx].cpu().numpy(), axs[i])
            show_point(points_1024[idx].numpy(), axs[i]) 
            axs[i].axis("off")
            axs[i].set_title(f"{names_temp[idx]} - Point: ({points_1024[idx][0]:.0f}, {points_1024[idx][1]:.0f})")
        
        plt.tight_layout()
        dataset_type = "validation" if is_validation else "training"
        plt.savefig(f"./data_sanitycheck_{dataset_type}.png", bbox_inches="tight", dpi=300)
        plt.close()
        
        print(f"Sanity check saved as: data_sanitycheck_{dataset_type}.png")
        break

print("=== SANITY CHECK FOR TRAINING DATASET ===")
sanity_check_dataset("data/BUSI", point_perturbation=40, is_validation=False)

print("\n=== SANITY CHECK FOR VALIDATION DATASET ===") 
sanity_check_dataset("data/BUSI", point_perturbation=40, is_validation=True)


def validate_model(model, val_dataloader, seg_loss, ce_loss, device, dice_metric):
    model.eval()
    val_loss = 0
    num_samples = 0 

    with torch.no_grad():
        for image, gt2D, points_1024, points_norm, _ in tqdm(val_dataloader, desc="Validation"):
            image, gt2D = image.to(device), gt2D.to(device).float()
            points_1024 = points_1024.to(device) 
            points_norm = points_norm.to(device) 
            pred = model(image, points_1024, points_norm)
            loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
            val_loss += loss.item()
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            dice_metric(y_pred=pred_binary, y=gt2D.float())
            
            num_samples += image.size(0)

    avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
    avg_dice_score = dice_metric.aggregate().item() if num_samples > 0 else 0.0
    dice_metric.reset()

    model.train()
    return avg_val_loss, avg_dice_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--tr_png_path",
        type=str,
        default="data/BUSI",
        help="path to training png files; two subfolders: images and masks",
    )
    parser.add_argument(
        "--val_png_path",
        type=str,
        default="data/BUSI",
        help="path to validation png files; two subfolders: images and masks",
    )
    parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B-PointAndNewBoxPredictor-UnfrozenPromptEncoder")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument(
        "-sam_checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to the pre-trained SAM checkpoint (for MedSAM model)."
    )
    parser.add_argument(
        "-box_predictor_checkpoint", type=str, 
        default="work_dir/BoxPredictor-Normalized-Rectangle-Centered-ViT-B-WithVal-20260529-0540/box_predictor_best.pth",
        help="Path to the trained BoxPredictor checkpoint to load and freeze."
    )
    parser.add_argument(
        "--load_pretrain", type=bool, default=True, help="load pretrain model"
    )
    parser.add_argument("-pretrain_model_path", type=str, default="")
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    parser.add_argument("-num_epochs", type=int, default=5)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-num_workers", type=int, default=0)
    parser.add_argument("-point_perturbation", type=int, default=40)
    parser.add_argument(
        "-enlarge_box_percentage", type=float, default=20.0,
        help="Percentage to enlarge the predicted bounding box (e.g., 30 for 30%). Set to 0 for no enlargement."
    )
    parser.add_argument(
        "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
    )
    parser.add_argument(
        "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
    )
    parser.add_argument(
        "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
    )
    parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
    parser.add_argument(
        "--resume", type=str, default="", help="Resuming training from checkpoint"
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--include_point", 
        action="store_true", 
        default=False,
        help="Whether to include the original point prompt along with the predicted box to MedSAM."
    )

    args = parser.parse_args()
    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_path": args.tr_png_path,
                "val_data_path": args.val_png_path,
                "model_type": args.model_type,
                "point_perturbation": args.point_perturbation,
                "box_predictor_checkpoint": args.box_predictor_checkpoint,
                "enlarge_box_percentage": args.enlarge_box_percentage,
            },
        )

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    device = torch.device(args.device)

    print(f"Loading MedSAM model type: {args.model_type} from {args.sam_checkpoint}")
    sam_model = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)

    box_predictor = BoxPredictor(
        image_embedding_dim=256,
        image_embedding_size=(64, 64)
    ).to(device)
    if os.path.exists(args.box_predictor_checkpoint):
        print(f"Loading BoxPredictor weights from {args.box_predictor_checkpoint}")
        box_predictor_checkpoint = torch.load(args.box_predictor_checkpoint, map_location="cpu")
        box_predictor.load_state_dict(box_predictor_checkpoint["model"])
        print("BoxPredictor weights loaded successfully.")
    else:
        print("Error: BoxPredictor checkpoint not found.")
        raise Exception

    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        box_predictor=box_predictor, 
        enlarge_box_percentage=args.enlarge_box_percentage, 
        include_point=args.include_point
    ).to(device)
    medsam_model.train() 

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.prompt_encoder.parameters() if p.requires_grad) +
        sum(p.numel() for p in medsam_model.mask_decoder.parameters() if p.requires_grad),
    )

    trainable_params = list(medsam_model.prompt_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    train_dataset = Dataset(args.tr_png_path, args.point_perturbation, is_validation=False)
    val_dataset = Dataset(args.val_png_path, args.point_perturbation, is_validation=True)
    
    print("Number of training samples: ", len(train_dataset))
    print("Number of validation samples: ", len(val_dataset))
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_epochs = args.num_epochs
    iter_num = 0
    train_losses = []
    val_losses = []
    val_dice_scores = []
    best_val_dice = -1.0 

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "best_val_dice" in checkpoint:
            best_val_dice = checkpoint["best_val_dice"]
    
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    print("Starting training MedSAM Mask Decoder and Prompt Encoder with Point and Box Prompts...")
    for epoch in range(start_epoch, num_epochs):
        medsam_model.train()
        epoch_loss = 0
        for step, (image, gt2D, points_1024, points_norm, _) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
            optimizer.zero_grad()
            image, gt2D = image.to(device), gt2D.to(device).float()
            points_1024 = points_1024.to(device) 
            points_norm = points_norm.to(device) 
            
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, points_1024, points_norm)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                medsam_pred = medsam_model(image, points_1024, points_norm)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            iter_num += 1

        avg_train_loss = epoch_loss / (step + 1)
        train_losses.append(avg_train_loss)
        val_loss, val_dice = validate_model(medsam_model, val_dataloader, seg_loss, ce_loss, device, dice_metric)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)

        if args.use_wandb:
            wandb.log({"train_loss": avg_train_loss, "val_loss": val_loss, "val_dice": val_dice, "epoch": epoch})

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_dice": best_val_dice,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint["best_val_dice"] = best_val_dice
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
            print(f"New best model saved with validation DICE: {best_val_dice:.4f}")

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(val_losses, label='Val Loss', color='orange')
        plt.title("Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(val_dice_scores, label='Val Dice', color='green')
        plt.title("Validation Dice Score")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(join(model_save_path, args.task_name + "_metrics.png"))
        plt.close()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Please ensure you have a CUDA-compatible GPU and PyTorch is installed correctly with CUDA support.")
        import sys
        sys.exit(1)
    
    main()