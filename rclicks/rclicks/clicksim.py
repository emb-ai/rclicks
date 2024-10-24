from pathlib import Path
import torch
import albumentations
import numpy as np
import typing as t
from easydict import EasyDict as edict
from .clickmap import Clicks
from . import data
from . import utils
import isegm
import cv2
from transalnet.utils.data_process import preprocess_img, postprocess_img
from transalnet.TranSalNet_Res import TranSalNet
from torchvision import transforms
toPIL = transforms.ToPILImage()


class SimClicks(Clicks):
    """
    Class for predicting click probabilty map and generating 
    clicks using click `model`.
    """
    @classmethod
    def load(
        cls,
        dataset_name,
        model: t.Callable,
        num_clicks=50,
        seed=None,
        files=None):
        
        dataset_obj = data.ISEGM_DATASETS[dataset_name]
        clicks_csv = Path(data.CLICKS_CSV) # needed only to load full_stems, clicks coordinates are not used
        prev_mask_dir = Path(data.PREV_MASK_PATHS[dataset_name])
        self = cls(dataset_name, 
                   dataset_obj=dataset_obj,
                   clicks_csv=clicks_csv,
                   prev_mask_dir=prev_mask_dir,
                   model=model,
                   num_clicks=num_clicks,
                   seed=seed,
                   files=files,
                )
        return self

    def __init__(
            self, 
            dataset_name,
            dataset_obj: isegm.data.base.ISDataset,
            clicks_csv: t.Union[Path, str], # needed only to load full_stems, clicks coordinates are not used
            prev_mask_dir: t.Union[Path, str],
            model: t.Callable,
            num_clicks=50,
            seed=None,
            files=None,
            ):
        
        self.model = model
        self.num_clicks = num_clicks
        self.seed = seed
        super().__init__(
            dataset_name=dataset_name, 
            dataset_obj=dataset_obj,
            clicks_csv=clicks_csv,
            prev_mask_dir=prev_mask_dir,
            files=files,
        )

    @property
    def device(self):
        return None

    @device.setter
    def device(self, val):
        raise RuntimeError("ClickModel's possible value for device is only None!")
    
    def _sample_from_dist(self, prob_map, num_points=50, seed=None):
        if seed is None:
            seed = self.seed
        if seed is not None:
            np.random.seed(seed)
        if num_points is None:
            num_points = self.num_clicks
        
        prob_map = prob_map / prob_map.sum()
        flat = prob_map.flatten()
        sample_index = np.random.choice(a=flat.size, p=flat, size=num_points)
        adjusted_index = np.unravel_index(sample_index, prob_map.shape)
        xs, ys, w, h = adjusted_index[1], adjusted_index[0], prob_map.shape[1], prob_map.shape[0] 
        return xs, ys, w, h

    def clicks(self, idx, num_points=None, seed=None):
        if num_points is None:
            num_points = self.num_clicks
        
        click_map = self.click_map(idx)
        res = self._sample_from_dist(
            click_map,
            num_points=num_points,
            seed=seed,
            )
        return res

    def click_map(self, idx, **kwargs):
        sample = self._dataset_sample(idx)
        img = sample.image / 255.
        encode = self.encode(idx)
        mask = self._mask(sample, encode)
        prev_mask = self.prev_mask(idx)
        click_type = self.click_type(idx)
        error_mask = self._error_mask(click_type, mask, prev_mask)
        return self.model(img, mask, error_mask)

    def sample(self, idx, **sample_kwargs):
        out = edict()
        out['full_stem'] = self.full_stem(idx)
        out['dataset'] = self.dataset_name
        out['click_type'] = self.click_type(idx)
        out['model_type'] = self.model_type(idx)
        sample = self._dataset_sample(idx)
        img = sample.image / 255.
        out['image'] = img
        out['encode'] = self.encode(idx)
        out['mask'] = self._mask(sample, out.encode)
        out['prev_mask'] = self.prev_mask(idx)
        out['error_mask'] = self._error_mask(out.click_type, out.mask, out.prev_mask)

        out['click_map'] = self.model(out.image, out.mask, out.error_mask)
        out['xs'], out['ys'], out['w'], out['h'] = self._sample_from_dist(
            out.click_map, **sample_kwargs)
        return out


def uniform_model(img, mask, error_mask):
    return error_mask / error_mask.max()

def distance_transform_model(img, mask, error_mask):
    dt = cv2.distanceTransform((error_mask*255).astype(np.uint8), cv2.DIST_L2, 0)
    return dt / dt.max()


class SaliencyModel():

    def __init__(self, sal_model, bbox_rsize=1.4):
        self.sal_model = sal_model
        self.bbox_rsize = bbox_rsize

    def get_frame_to_cut(self, mask, coef):
        rows, cols = np.where(mask == 1)
        
        if len(rows) == 0 or len(cols) == 0:
            return 0, 0
        
        # Calculate the height and width
        height = np.max(rows) - np.min(rows) + 1
        new_height = int(coef * height)
        width = np.max(cols) - np.min(cols) + 1
        new_width = int(coef * width)
        dif_height = int((new_height - height) / 2)
        dif_width = int((new_width - width) / 2)
        return (np.clip(np.min(rows) - dif_height, 0, mask.shape[0]),
                np.clip(np.max(rows) + dif_height, 0, mask.shape[0]),
                np.clip(np.min(cols) - dif_width, 0, mask.shape[1]),
                np.clip(np.max(cols) + dif_width, 0, mask.shape[1]))

    def cut_image_from_mask(self, image, mask):
        min_row, max_row, min_col, max_col = self.get_frame_to_cut(
            mask, self.bbox_rsize)
        x1, y1 = min_row, min_col  # Top-left corner coordinates
        x2, y2 = max_row + 1, max_col + 1  # Bottom-right corner coordinates
        image = image[x1:x2, y1:y2]
        msk = mask[x1:x2, y1:y2]
        mask = (msk * 255).astype(np.uint8)
        grey_image = np.zeros_like(image)
        grey_image[:] = (0.5, 0.5, 0.5)
        result = cv2.bitwise_and(image, image, mask=mask)
        inverted_mask = cv2.bitwise_not(mask)
        grey_outside = cv2.bitwise_and(grey_image, grey_image, mask=inverted_mask)
        final_result = cv2.add(result, grey_outside)
        return final_result, (x1, y1, x2, y2), msk

    def paste_saliency_on_image(self, pred_saliency, msk, coords):
        msk = np.zeros_like(msk)
        x1, y1, x2, y2 = coords
        msk[x1:x2, y1:y2] = pred_saliency
        return msk

    def __call__(self, img, mask, error_mask):
        msk = error_mask
        img, coords, msk_pst = self.cut_image_from_mask(img, msk)
        img_pst = img      
        
        img = preprocess_img((img*255.).astype(np.uint8))
        img = np.array(img) / 255.
        img = np.expand_dims(np.transpose(img,(2,0,1)),axis=0)
        img = torch.from_numpy(img)
        img = img.type(torch.cuda.FloatTensor).cuda()
        
        pred_saliency = self.sal_model(img)
        
        pic = toPIL(pred_saliency.squeeze())
        pred_saliency = (postprocess_img(pic, img_pst)).astype(float)
        pred_saliency *= msk_pst
        if np.sum(pred_saliency) == 0:
            pred_saliency = msk_pst
        else: 
            pred_saliency = pred_saliency / np.sum(pred_saliency)
        
        pred_saliency = self.paste_saliency_on_image(pred_saliency, msk, coords)
        
        return pred_saliency


class SaliencyImageModel(SaliencyModel):
    def cut_image_from_mask(self, image, mask):
        min_row, max_row, min_col, max_col = self.get_frame_to_cut(
            mask, self.bbox_rsize)
        x1, y1 = min_row, min_col  # Top-left corner coordinates
        x2, y2 = max_row + 1, max_col + 1  # Bottom-right corner coordinates
        image = image[x1:x2, y1:y2]
        msk = mask[x1:x2, y1:y2]
        return image, (x1, y1, x2, y2), msk


class ClickabilityModel():

    @classmethod
    def load(cls, model_fname):
        model = utils.load_segnext(model_fname)
        return cls(
            model
        )

    def __init__(self, model):
        self.model = model
        self.normalization = albumentations.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225], 
            max_pixel_value=1.0)


    def _input_for_inference(self, img, mask, error_mask):
        frame = self.normalization(image=img)['image']
        frame = utils.padding(frame)
        mask = utils.padding(mask)
        error_mask = utils.padding(error_mask)
        return (
            torch.from_numpy(np.transpose(frame[None, :, :, :], (0, 3, 1, 2))),
            torch.from_numpy(mask[None, None, :, :]),
            torch.from_numpy(error_mask[None, None, :, :]),
            None,
        )

    def __call__(self, img, mask, error_mask):
        frame, mask, error_mask, _  = self._input_for_inference(
            img, mask, error_mask)
        frame = frame.cuda()
        mask = mask.cuda()
        error_mask = error_mask.cuda()
        coord_feature = torch.cat((mask, error_mask), 1)
        pred = self.model(frame, coord_feature)['instances']
        click_map = utils.unpadding(pred, img.shape)
        click_map = click_map[0, 0, :, :].cpu().detach().numpy()
        click_map = (click_map - np.min(click_map)) / (np.max(click_map) - np.min(click_map))
        return click_map
