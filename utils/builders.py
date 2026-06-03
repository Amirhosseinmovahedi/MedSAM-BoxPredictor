from models.box_predictor import BoxPredictor
from models.medsam import MedSAMWithBox, MedSAMWithPoint
from datasets import Dataset

def build_box_predictor(image_embedding_dim=256, image_embedding_size=(64, 64), patch_size=5):
    return BoxPredictor(image_embedding_dim, image_embedding_size, patch_size=5)

def build_medsam(model_type, image_encoder, mask_decoder, prompt_encoder, box_predictor=None, enlarge_box_percentage=25.0, use_point=False):
    if model_type == 1:
        if box_predictor == None:
            raise ValueError("The box_predictor module is None")
        return MedSAMWithBox(image_encoder, mask_decoder, prompt_encoder, box_predictor, enlarge_box_percentage, use_point)
    elif model_type == 2:
        return MedSAMWithPoint(image_encoder, mask_decoder, prompt_encoder)
    else:
        raise ValueError(f"Unknown MedSAM version: {model_type}")
    
def build_dataset(data_root, point_perturbation=40):
    return Dataset(data_root, point_perturbation)