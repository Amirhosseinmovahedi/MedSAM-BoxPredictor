import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 1024

class BoxPredictor(nn.Module):
    def __init__(self,
                 image_embedding_dim=256, 
                 image_embedding_size=(64, 64), 
                 patch_size=5 
                ):
        super().__init__()
        self.image_embedding_dim = image_embedding_dim
        self.image_embedding_size = image_embedding_size
        self.patch_size = patch_size

        input_mlp_dim = (patch_size * patch_size * image_embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_mlp_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, image_embeddings, point_coords_normalized):
        B, C, H_emb, W_emb = image_embeddings.shape

        if point_coords_normalized.dim() == 3 and point_coords_normalized.shape[1] == 1:
            point_coords_normalized = point_coords_normalized.squeeze(1)

        point_coords_emb_x = (point_coords_normalized[:, 0] * W_emb).long()
        point_coords_emb_y = (point_coords_normalized[:, 1] * H_emb).long()

        half_patch_side = self.patch_size // 2
        
        batch_features = []
        for i in range(B):
            px_emb = point_coords_emb_x[i]
            py_emb = point_coords_emb_y[i]

            x_start = max(0, px_emb - half_patch_side)
            x_end = min(W_emb, px_emb + half_patch_side + 1)
            y_start = max(0, py_emb - half_patch_side)
            y_end = min(H_emb, py_emb + half_patch_side + 1)

            patch = image_embeddings[i, :, y_start:y_end, x_start:x_end]

            pad_left = max(0, half_patch_side - px_emb)
            pad_right = max(0, (px_emb + half_patch_side + 1) - W_emb)
            pad_top = max(0, half_patch_side - py_emb)
            pad_bottom = max(0, (py_emb + half_patch_side + 1) - H_emb)
            
            patch = F.pad(patch, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

            patch = patch[:, :self.patch_size, :self.patch_size] 

            batch_features.append(patch.flatten()) 

        patch_features = torch.stack(batch_features, dim=0) 

        predicted_raw_params = self.mlp(patch_features)

        predicted_width_raw = predicted_raw_params[:, 0]
        predicted_height_raw = predicted_raw_params[:, 1]
        predicted_offset_x_norm = predicted_raw_params[:, 2] 
        predicted_offset_y_norm = predicted_raw_params[:, 3]

        predicted_width_norm = F.softplus(predicted_width_raw)
        predicted_height_norm = F.softplus(predicted_height_raw)

        center_x_initial_norm = point_coords_normalized[:, 0]
        center_y_initial_norm = point_coords_normalized[:, 1]

        new_center_x_norm = center_x_initial_norm + predicted_offset_x_norm
        new_center_y_norm = center_y_initial_norm + predicted_offset_y_norm

        x_min_norm = new_center_x_norm - predicted_width_norm / 2
        y_min_norm = new_center_y_norm - predicted_height_norm / 2
        x_max_norm = new_center_x_norm + predicted_width_norm / 2
        y_max_norm = new_center_y_norm + predicted_height_norm / 2

        x_min_norm = torch.clamp(x_min_norm, 0.0, 1.0)
        y_min_norm = torch.clamp(y_min_norm, 0.0, 1.0)
        x_max_norm = torch.clamp(x_max_norm, 0.0, 1.0)
        y_max_norm = torch.clamp(y_max_norm, 0.0, 1.0)

        min_dim_norm = 1.0 / IMAGE_SIZE 
        x_max_norm = torch.max(x_max_norm, x_min_norm + min_dim_norm) 
        y_max_norm = torch.max(y_max_norm, y_min_norm + min_dim_norm) 

        predicted_box_norm = torch.stack([x_min_norm, y_min_norm, x_max_norm, y_max_norm], dim=1) 

        return predicted_box_norm