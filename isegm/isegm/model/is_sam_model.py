import torch
import torch.nn as nn
import numpy as np
import cv2
from isegm.model.modifiers import LRMult
from isegm.inference import utils
from copy import deepcopy
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class ISModelSAM(nn.Module):
    def __init__(self, device='cuda', model_path=None):
        super().__init__()
        model_type = 'vit_b' if 'vit_b' in str(model_path) else 'vit_h' if 'vit_h' in str(model_path) else 'vit_l'
        sam = sam_model_registry[model_type](checkpoint=model_path)
        for n, p in sam.named_parameters():
            p.requires_grad = False
        sam.eval()
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        self.prev_mask = None
        self.resize = ResizeLongestSide(sam.image_encoder.img_size)
        self.with_prev_mask = True
        self.binary_prev_mask = False

    
    def forward(self, image, points):
        
        points = points.cpu().numpy()
        
        if points.shape[1] == 2:
            self.prev_mask = None
        
        image, prev_mask = self.prepare_input(image)

        input_image = self.resize.apply_image_torch(image * 255)
           
        self.sam_predictor.set_torch_image(input_image, image.shape[2:])

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
        input_label = list(all_list[:, 0])

        old_h, old_w = image.shape[2:]
        new_h, new_w = get_preprocess_shape(old_h, old_w, self.sam_predictor.transform.target_length)

        points_list = points_list.astype(float)
        points_list[..., 0] = points_list[..., 0] * (new_w / old_w)
        points_list[..., 1] = points_list[..., 1] * (new_h / old_h)
        points_list = torch.as_tensor(points_list, dtype=torch.float, device=self.sam_predictor.device)[None]
        input_label = torch.tensor(input_label)[None].to(self.sam_predictor.device)

        res, scores, logits = self.sam_predictor.predict_torch(
                    point_coords=points_list,
                    point_labels=input_label,
                    mask_input=self.prev_mask,
                    multimask_output=True,
                    return_logits=True)

        prediction = torch.sigmoid(res)
        prediction = prediction[0, torch.argmax(scores)][None, None]

        # Since SAM use its own prev mask format
        self.prev_mask = logits[0, torch.argmax(scores)][None, None]
        
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


    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features