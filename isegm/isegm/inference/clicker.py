import numpy as np
from copy import deepcopy
import cv2
import torch


class Clicker(object):
    def __init__(self, gt_mask=None, image=None, init_clicks=None, ignore_label=-1, click_indx_offset=0, 
                 click_model=None, model_device='cuda', quantile_low=0.0, quantile_high=1.0):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        
        self.image = image
        self.reset_clicks()
        
        self.click_model = click_model

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask)
        self.add_click(click)
        return self.click_map, click
    
    def make_next_click_coords(self, coords):

        coords[1] = np.clip(coords[1], 0, self.image.shape[0] - 1)
        coords[0] = np.clip(coords[0], 0, self.image.shape[1] - 1)
        
        click = Click(is_positive=True, coords=(coords[1], coords[0]))
        self.add_click(click)
        return None

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]
    

    def _get_next_click(self, pred_mask, padding=True):
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist

        if is_positive:
            if self.click_model is not None:

                click_map = self.click_model.apply(self.image, self.gt_mask, fn_mask[1:-1, 1:-1])[0][0].detach().cpu().numpy()
                self.click_map = click_map * fn_mask[1:-1, 1:-1].astype(np.float32) * self.not_clicked_map

                sorted_clickmap = np.sort(self.click_map.flatten())
                quantile_low = sorted_clickmap.sum() * self.quantile_low
                quantile_high = sorted_clickmap.sum() * self.quantile_high
                
                idx_low = sorted_clickmap.cumsum().searchsorted(quantile_low, side='left')
                idx_high = sorted_clickmap.cumsum().searchsorted(quantile_high, side='right')
                if idx_high == len(sorted_clickmap):
                    idx_high -= 1
                
                thr_low = sorted_clickmap[idx_low]
                thr_high = sorted_clickmap[idx_high]
                self.click_map = self.click_map * (self.click_map >= thr_low) * (self.click_map <= thr_high)

                probs = self.click_map / np.sum(self.click_map)
                
                if np.isnan(probs).any():
                    self.click_map = fn_mask_dt
                    coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)
                    coords_y = coords_y[0]
                    coords_x = coords_x[0]
                else:
                    coords_y, coords_x = np.unravel_index(np.random.choice(np.arange(probs.size), p=probs.flatten()), probs.shape)

            else:
                self.click_map = fn_mask_dt
                coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)
                coords_y = coords_y[0]
                coords_x = coords_x[0]

        else:

            if self.click_model is not None:
                click_map = self.click_model.apply(self.image, self.gt_mask, fp_mask[1:-1, 1:-1])[0][0].detach().cpu().numpy()
                self.click_map = click_map * fp_mask[1:-1, 1:-1].astype(np.float32) * self.not_clicked_map

                sorted_clickmap = np.sort(self.click_map.flatten())
                quantile_low = sorted_clickmap.sum() * self.quantile_low
                quantile_high = sorted_clickmap.sum() * self.quantile_high
                
                idx_low = sorted_clickmap.cumsum().searchsorted(quantile_low, side='left')
                idx_high = sorted_clickmap.cumsum().searchsorted(quantile_high, side='right')
                if idx_high == len(sorted_clickmap):
                    idx_high -= 1
                
                thr_low = sorted_clickmap[idx_low]
                thr_high = sorted_clickmap[idx_high]
                self.click_map = self.click_map * (self.click_map >= thr_low) * (self.click_map <= thr_high)

                probs = self.click_map / np.sum(self.click_map)
                    
                if np.isnan(probs).any():
                    self.click_map = fp_mask_dt
                    coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)
                    coords_y = coords_y[0]
                    coords_x = coords_x[0]
                else:
                    coords_y, coords_x = np.unravel_index(np.random.choice(np.arange(probs.size), p=probs.flatten()), probs.shape)

            else:
                self.click_map = fp_mask_dt
                coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)
                coords_y = coords_y[0]
                coords_x = coords_x[0]

        return Click(is_positive=is_positive, coords=(coords_y, coords_x))
        

    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            if self.gt_mask is not None:
                self.not_clicked_map[min(round(coords[0]), self.not_clicked_map.shape[0] - 1), 
                                     min(round(coords[1]), self.not_clicked_map.shape[1] - 1)] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[min(round(coords[0]), self.not_clicked_map.shape[0] - 1), 
                                 min(round(coords[1]), self.not_clicked_map.shape[1] - 1)] = True

        return click

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class Click:
    def __init__(self, is_positive, coords, indx=None):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy
