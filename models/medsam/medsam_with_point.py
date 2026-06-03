import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 1024

class MedSAMWithPoint(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, point):
        with torch.no_grad():
            image_embedding = self.image_encoder(image) 

            point_torch = torch.as_tensor(point, dtype=torch.float32, device=image.device)
            if len(point_torch.shape) == 2:
                point_torch = point_torch[:, None, :]

            point_labels = torch.ones(point_torch.shape[0], point_torch.shape[1], device=image.device)
            
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_torch, point_labels),
                boxes=None,
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