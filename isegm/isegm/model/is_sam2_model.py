import torch
import torch.nn as nn
import numpy as np
import cv2
from copy import deepcopy
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ISModelSAM2(nn.Module):
    def __init__(self, device='cuda', model_path=None):
        super().__init__()
        
        model_type = 'sam2_hiera_large' if 'sam2_hiera_large' in str(model_path) else 'sam2_hiera_base_plus' if 'sam2_hiera_base_plus' in str(model_path) else 'sam2_hiera_small' if 'sam2_hiera_small' in str(model_path) else 'sam2_hiera_tiny' if 'sam2_hiera_tiny' in str(model_path) else 'sam2.1_hiera_large' if 'sam2.1_hiera_large' in str(model_path) else 'sam2.1_hiera_base_plus' if 'sam2.1_hiera_base_plus' in str(model_path) else 'sam2.1_hiera_small' if 'sam2.1_hiera_small' in str(model_path) else 'sam2.1_hiera_tiny'
    
        if model_type == 'sam2_hiera_large':
            model_cfg = 'configs/sam2/sam2_hiera_l.yaml'
        elif model_type == 'sam2_hiera_base_plus':
            model_cfg = 'configs/sam2/sam2_hiera_b+.yaml'
        elif model_type == 'sam2_hiera_small':
            model_cfg = 'configs/sam2/sam2_hiera_s.yaml'
        elif model_type == 'sam2_hiera_tiny':
            model_cfg = 'configs/sam2/sam2_hiera_t.yaml'
        elif model_type == 'sam2.1_hiera_large':
            model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
        elif model_type == 'sam2.1_hiera_base_plus':
            model_cfg = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
        elif model_type == 'sam2.1_hiera_small':
            model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'
        elif model_type == 'sam2.1_hiera_tiny':
            model_cfg = 'configs/sam2.1/sam2.1_hiera_t.yaml'
        
        sam2 = build_sam2(model_cfg, model_path)
        for n, p in sam2.named_parameters():
            p.requires_grad = False
        sam2.eval()
        sam2.to(device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2)
        self.prev_mask = None
        self.with_prev_mask = True
        self.binary_prev_mask = False

    
    def forward(self, image, points):
        
        points = points.cpu().numpy()
        
        if points.shape[1] == 2:
            self.prev_mask = None
        
        image, prev_mask = self.prepare_input(image)
        input_image = (image.cpu().numpy() * 255)[0].transpose(1, 2, 0).astype(np.uint8)

        self.sam2_predictor.set_image(input_image)

        points_list = []
        for idx in range(points.shape[1]):
            if points[0][idx][-1] < 0:
                continue
            if idx >= points.shape[1] // 2:
                # Negative click
                points_list.append(np.hstack([0, points[0][idx]]))
            else:
                points_list.append(np.hstack([1, points[0][idx]]))

        points_list = np.array(points_list)
        all_list = points_list[points_list[:, 3].argsort()]
        points_list = all_list[:, 1:3][:, ::-1].copy()
        input_label = np.array(list(all_list[:, 0]))
        
        res, scores, logits = self.sam2_predictor.predict(
                    point_coords=points_list,
                    point_labels=input_label,
                    mask_input=self.prev_mask,
                    multimask_output=False,
                    return_logits=True)
        
        prediction = torch.sigmoid(torch.tensor(res)[None].to(image.device))
        
        # Since SAM use its own prev mask format
        self.prev_mask = logits[np.argmax(scores), :, :][None, :, :]
        
        outputs = {'instances':  prediction}
        
        return outputs


    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()
        return image, prev_mask