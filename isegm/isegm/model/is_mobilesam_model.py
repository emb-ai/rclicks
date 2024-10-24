import torch
from mobile_sam import SamPredictor, sam_model_registry
from mobile_sam.utils.transforms import ResizeLongestSide
from .is_sam_model import ISModelSAM


class ISModelMobileSAM(ISModelSAM):
    def __init__(self, device='cuda', model_path=None):
        super().__init__()
        model_type = 'vit_t'
        mobile_sam = sam_model_registry[model_type](checkpoint=model_path)
        for n, p in mobile_sam.named_parameters():
            p.requires_grad = False
        mobile_sam.eval()
        mobile_sam.to(device=device)
        self.sam_predictor = SamPredictor(mobile_sam)
        self.prev_mask = None
        self.resize = ResizeLongestSide(mobile_sam.image_encoder.img_size)
        self.with_prev_mask = True
        self.binary_prev_mask = False