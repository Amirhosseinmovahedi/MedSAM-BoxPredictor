import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
import argparse
from datetime import datetime
import shutil
from utils.functions import generalized_iou_loss
from models.box_predictor import BoxPredictor
from datasets.BoxPredictorDataset import BoxPredictorDataset

try:
    from torchvision.ops.boxes import box_iou
    print("torchvision.ops.boxes.box_iou imported successfully.")
except ImportError:
    print("Warning: torchvision.ops.boxes.box_iou not found. IoU calculation will use a placeholder.")
    def box_iou(boxes1, boxes2):
        print("Error: box_iou is a placeholder. Please install torchvision (pip install torchvision).")
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)


torch.manual_seed(2023)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

IMAGE_SIZE = 1024.0 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--data_root_path",
        type=str,
        default="data/BUSI",
        help="path to data root; expects imgs, gts, imgs_val, gts_val subfolders",
    )
    parser.add_argument("-task_name", type=str, default="BoxPredictor-Normalized-Rectangle-Centered-ViT-B-WithVal")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-perturbation", type=int, default=40)
    parser.add_argument(
        "-checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to the pre-trained SAM checkpoint (for image encoder)."
    )
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    parser.add_argument("-num_epochs", type=int, default=10)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-num_workers", type=int, default=2)
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
    args = parser.parse_args()

    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_root_path": args.data_root_path,
                "model_type": args.model_type,
                "num_epochs": args.num_epochs,
                "weight_decay": args.weight_decay,
            },
        )

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    device = torch.device(args.device)
    print(f"Loading SAM model type: {args.model_type} from {args.checkpoint}")
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

    image_encoder = sam_model.image_encoder.to(device)
    for param in image_encoder.parameters():
        param.requires_grad = False
    image_encoder.eval() 

    box_predictor = BoxPredictor(
        image_embedding_dim=256, 
        image_embedding_size=(64, 64)
    ).to(device)
    box_predictor.train()

    print(
        "Number of total parameters in BoxPredictor: ",
        sum(p.numel() for p in box_predictor.parameters()),
    )

    optimizer = torch.optim.AdamW(
        box_predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of trainable parameters (BoxPredictor only): ",
        sum(p.numel() for p in box_predictor.parameters() if p.requires_grad),
    )

    box_l1_loss = nn.L1Loss(reduction="mean")

    train_dataset = BoxPredictorDataset(args.data_root_path, img_subdir="imgs", gt_subdir="gts", perturbation=args.perturbation)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    try:
        val_dataset = BoxPredictorDataset(args.data_root_path, img_subdir="imgs_val", gt_subdir="gts_val", perturbation=args.perturbation)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
        )
        has_validation = True
        print("Validation dataset loaded successfully.")
    except FileNotFoundError as e:
        print(f"Validation data not found: {e}. Skipping validation rounds.")
        has_validation = False


    num_epochs = args.num_epochs
    iter_num = 0
    train_losses = [] 
    best_val_loss = 1e10
    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            box_predictor.load_state_dict(checkpoint["model"]) 
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "best_val_loss" in checkpoint: 
                best_val_loss = checkpoint["best_val_loss"]
            if "train_losses" in checkpoint: 
                train_losses = checkpoint["train_losses"]
        else:
            print(f"Checkpoint file not found at {args.resume}. Starting training from scratch.")

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    print("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        box_predictor.train()
        epoch_total_train_box_loss = 0
        epoch_train_l1_loss = 0
        epoch_train_giou_loss = 0

        for step, (image, _, points_norm, gt_bbox_norm, _) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")):
            optimizer.zero_grad()
            
            image, points_norm, gt_bbox_norm = image.to(device), points_norm.to(device), gt_bbox_norm.to(device)

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    with torch.no_grad(): 
                        image_embedding = image_encoder(image) 

                    predicted_box_norm = box_predictor(image_embedding, points_norm)

                    current_l1_loss = box_l1_loss(predicted_box_norm, gt_bbox_norm)
                    current_giou_loss = generalized_iou_loss(predicted_box_norm, gt_bbox_norm)
                    
                    total_box_loss = current_l1_loss + current_giou_loss
                
                scaler.scale(total_box_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad(): 
                    image_embedding = image_encoder(image) # (B, 256, 64, 64)

                predicted_box_norm = box_predictor(image_embedding, points_norm) # (B, 4)
                current_l1_loss = box_l1_loss(predicted_box_norm, gt_bbox_norm)
                current_giou_loss = generalized_iou_loss(predicted_box_norm, gt_bbox_norm)
                
                total_box_loss = current_l1_loss + current_giou_loss
                
                total_box_loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            epoch_total_train_box_loss += total_box_loss.item()
            epoch_train_l1_loss += current_l1_loss.item()
            epoch_train_giou_loss += current_giou_loss.item()
            iter_num += 1

        avg_epoch_total_train_box_loss = epoch_total_train_box_loss / (step + 1)
        avg_epoch_train_l1_loss = epoch_train_l1_loss / (step + 1)
        avg_epoch_train_giou_loss = epoch_train_giou_loss / (step + 1)
        
        train_losses.append(avg_epoch_total_train_box_loss)
        if args.use_wandb:
            wandb.log({
                "epoch_total_train_box_loss": avg_epoch_total_train_box_loss,
                "epoch_train_l1_loss": avg_epoch_train_l1_loss,
                "epoch_train_giou_loss": avg_epoch_train_giou_loss,
                "epoch": epoch
            })
        
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch+1}/{num_epochs}, '
            f'Train Total Box Loss: {avg_epoch_total_train_box_loss:.4f}, Train L1 Loss: {avg_epoch_train_l1_loss:.4f}, '
            f'Train GIoU Loss: {avg_epoch_train_giou_loss:.4f}'
        )

        if has_validation:
            box_predictor.eval()
            epoch_total_val_box_loss = 0
            epoch_val_l1_loss = 0
            epoch_val_giou_loss = 0

            with torch.no_grad():
                for val_step, (image, _, points_norm, gt_bbox_norm, _) in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation")):
                    image, points_norm, gt_bbox_norm = image.to(device), points_norm.to(device), gt_bbox_norm.to(device)

                    image_embedding = image_encoder(image)
                    predicted_box_norm = box_predictor(image_embedding, points_norm)

                    current_l1_loss = box_l1_loss(predicted_box_norm, gt_bbox_norm)
                    current_giou_loss = generalized_iou_loss(predicted_box_norm, gt_bbox_norm)
                    
                    total_box_loss = current_l1_loss + current_giou_loss

                    epoch_total_val_box_loss += total_box_loss.item()
                    epoch_val_l1_loss += current_l1_loss.item()
                    epoch_val_giou_loss += current_giou_loss.item()

            avg_epoch_total_val_box_loss = epoch_total_val_box_loss / (val_step + 1)
            avg_epoch_val_l1_loss = epoch_val_l1_loss / (val_step + 1)
            avg_epoch_val_giou_loss = epoch_val_giou_loss / (val_step + 1)

            if args.use_wandb:
                wandb.log({
                    "epoch_total_val_box_loss": avg_epoch_total_val_box_loss,
                    "epoch_val_l1_loss": avg_epoch_val_l1_loss,
                    "epoch_val_giou_loss": avg_epoch_val_giou_loss,
                    "epoch": epoch
                })
            
            print(
                f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch+1}/{num_epochs}, '
                f'Val Total Box Loss: {avg_epoch_total_val_box_loss:.4f}, Val L1 Loss: {avg_epoch_val_l1_loss:.4f}, '
                f'Val GIoU Loss: {avg_epoch_val_giou_loss:.4f}'
            )

            if avg_epoch_total_val_box_loss < best_val_loss:
                best_val_loss = avg_epoch_total_val_box_loss
                checkpoint = {
                    "model": box_predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "train_losses": train_losses
                }
                torch.save(checkpoint, join(model_save_path, "box_predictor_best.pth"))
                print(f"Saved best BoxPredictor model with Val Loss: {best_val_loss:.4f}")
        else:
            if avg_epoch_total_train_box_loss < best_val_loss:
                best_val_loss = avg_epoch_total_train_box_loss
                checkpoint = {
                    "model": box_predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss, 
                    "train_losses": train_losses
                }
                torch.save(checkpoint, join(model_save_path, "box_predictor_best.pth"))
                print(f"Saved best BoxPredictor model with Train Loss: {best_val_loss:.4f} (No validation data)")

        checkpoint = {
            "model": box_predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses
        }
        torch.save(checkpoint, join(model_save_path, "box_predictor_latest.pth"))
        plt.figure()
        plt.plot(train_losses)
        plt.title("Box Predictor Training Loss (L1 + GIoU)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(join(model_save_path, args.task_name + "_train_loss.png"))
        plt.close()

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()