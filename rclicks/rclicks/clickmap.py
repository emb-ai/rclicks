from pathlib import Path
import cv2
from easydict import EasyDict as edict
import numpy as np
from scipy.ndimage import gaussian_filter
import typing as t
import isegm
from . import data
from . import utils
import pandas as pd
from easydict import EasyDict as edict


def _image_with_object_stem_impl(full_stem: str):
        img_stem = full_stem.split(
            '_ritm')[0].split(
                '_sam')[0].split(
                    '_simple')[0].split(
                        '_mivos')[0]
        return img_stem

def _image_stem(full_stem: str):
    return _image_with_object_stem_impl(full_stem).split('_______')[0]

def _limited_image_size(width, height, max_side_size):
    if max_side_size is None:
        return width, height
    if height > width:
        target_h = min(height, max_side_size)
        target_w = int(width / height * target_h + 0.5)
    else:
        target_w = min(width, max_side_size)
        target_h = int(height / width * target_w + 0.5)
    return target_w, target_h

class Clicks():

    @classmethod
    def load(cls, dataset_name, files=None):
        dataset_obj = data.ISEGM_DATASETS[dataset_name]
        clicks_csv = Path(data.CLICKS_CSV)
        prev_mask_dir = Path(data.PREV_MASK_PATHS[dataset_name])
        return cls(dataset_name, 
                   dataset_obj=dataset_obj,
                   clicks_csv=clicks_csv,
                   prev_mask_dir=prev_mask_dir,
                   files=files)

    def __init__(
            self,
            dataset_name,
            dataset_obj: isegm.data.base.ISDataset,
            clicks_csv: t.Union[Path, str],
            prev_mask_dir: t.Union[Path, str],
            files=None,):
        self._init_dataset(dataset_name, dataset_obj, prev_mask_dir)
        self._init_files(files)
        self._init_clicks_df(clicks_csv)
        self._post_init()
        self._device = None
    
    def _init_dataset(self, dataset_name, dataset_obj, prev_mask_dir):
        self.dataset_name = dataset_name
        self.dataset = dataset_obj
        self.dataset_path = self.dataset.dataset_path
        self.dataset.dataset_samples = [
            imname 
            for imname in self.dataset.dataset_samples
            if imname != '.ipynb_checkpoints'
        ]
        self.prev_mask_dir = Path(prev_mask_dir)
    
    def _init_files(self, files):
        if files is not None:
            with open(files, 'r') as f:
                self.files = f.readlines()
            self.files = [i[:-1] for i in self.files]
        else:
            self.files = None
        
    def _init_clicks_df(self, clicks_csv):
        self._clicks_csv = clicks_csv
        self.clicks_df = pd.read_csv(clicks_csv, keep_default_na=False)
        self.clicks_df = self.clicks_df[self.clicks_df.dataset==self.dataset_name]
        if self.files is not None:
            self.clicks_df = self.clicks_df[self.clicks_df.image_stem.isin(self.files)]

    def _post_init(self):
        self.image_indxs_in_dataset = {
            Path(imname).stem: i for i, imname in enumerate(self.dataset.dataset_samples)}
        self.full_stems_indxs_in_dataset = dict()
        for full_stem in self.clicks_df.full_stem.unique().tolist():
            img_stem = _image_stem(full_stem)
            if img_stem in self.image_indxs_in_dataset and full_stem not in self.full_stems_indxs_in_dataset:
                self.full_stems_indxs_in_dataset[full_stem] = self.image_indxs_in_dataset[img_stem]
        
        self.full_stems = list(self.full_stems_indxs_in_dataset.keys())
    
    def __len__(self):
        return len(self.full_stems)
    
    def full_stem(self, idx) -> str:
        """
        Get unique (image, object, click_type, is_model_type) string index obtained from clicks dataframe.
        This index is used to obtain clicks from dataframe.
        """
        return self.full_stems[idx]

    def image_with_object_stem(self, idx):
        return _image_with_object_stem_impl(self.full_stem(idx))

    def image_stem(self, idx):
        return _image_stem(self.full_stem(idx))

    def _dataset_image_idx(self, idx):
        return self.full_stems_indxs_in_dataset[self.full_stem(idx)]

    def image_path(self, idx):
        sample_idx = self._dataset_image_idx(idx)
        return Path(self.dataset.dataset_path) / self.dataset._images_path / self.dataset.dataset_samples[sample_idx]
    
    def _dataset_sample(self, idx):
        sample_idx = self._dataset_image_idx(idx)
        sample = self.dataset.get_sample(sample_idx)
        return sample 
    
    def image(self, idx):
        sample = self._dataset_sample(idx)
        return sample.image
    
    @property
    def fp(self):
        """
        Return the false positive clicks for the subsequent round
        """
        return self.clicks_df[self.clicks_df.click_type=='fp']
    
    @property
    def fn(self):
        """
        Return the false negative clicks for the subsequent round
        """
        return self.clicks_df[self.clicks_df.click_type=='fn']
    
    @property
    def first(self):
        """
        Return the list of first clicks
        """
        return self.clicks_df[self.clicks_df.click_type=='first']
    
    def img_stem_to_full_stems(self, img_stem) -> list[str]:
        return [s for s in self.full_stems if img_stem in s]

    def indx(self, full_stem):
        return self.full_stems.index(full_stem)

    def click_type(self, idx) -> t.Union[t.Literal['fn'], t.Literal['fp'], t.Literal['first']]:
        full_stem = self.full_stem(idx)
        return self.clicks_df[self.clicks_df.full_stem == full_stem].click_type.to_list()[0]
    
    def model_type(self, idx) -> t.Union[str, None]:
        """
        Returns a model's name, that was used in the first round.
        For the first round clicks return None.
        """
        full_stem = self.full_stem(idx)
        model_type = self.clicks_df[self.clicks_df.full_stem == full_stem].model_type.to_list()[0]
        if model_type == '':
            return None
        return model_type
    
    def encode(self, idx) -> t.Union[int, None]:
        """
        Get object code for given index.
        """
        full_stem = self.full_stem(idx)
        object_stem = self.clicks_df[self.clicks_df.full_stem == full_stem].object_stem.to_list()[0]
        if object_stem == '':
            return None
        return int(object_stem)
    
    def prev_mask_path(self, idx) -> Path:
        full_stem = self.full_stem(idx)
        click_type: t.Union[t.Literal['fn'], t.Literal['fp'], t.Literal['first']] = self.click_type(idx)
        if click_type == 'first':
            return None
        path: Path = self.prev_mask_dir / f'{full_stem}.png'
        assert path.is_file()
        return path
    
    def prev_mask(self, idx):
        if self.prev_mask_path(idx) is None:
            return None
        prev_mask_path = self.prev_mask_path(idx)
        assert prev_mask_path.is_file()
        fname = str(prev_mask_path)
        prev_mask = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB).astype(np.float32)[:, :, 0] / 255.
        return prev_mask

    def _mask(self, sample, encode):
        if encode is None:
            return np.max(sample._encoded_masks > 0, axis=-1).astype(np.float32)
        return np.max(sample._encoded_masks == encode, axis=-1).astype(np.float32)

    def mask(self, idx):
        sample = self._dataset_sample(idx)
        encode = self.encode(idx)
        return self._mask(sample, encode)
    
    def _error_mask(self, click_type, mask, prev_mask):
        if click_type == 'first':
            error_mask = mask[:]
            return error_mask
        elif click_type == 'fn':
            neg_prev = prev_mask[:]
            neg_prev = 1 - neg_prev
            error_mask = mask * neg_prev
            return error_mask
        else:
            pos_mask = mask[:]
            pos_mask = 1 - pos_mask
            error_mask = pos_mask * prev_mask
            return error_mask
    
    def error_mask(self, idx):
        mask = self.mask(idx)
        prev_mask = self.prev_mask(idx)
        click_type: t.Union[t.Literal['fn'], t.Literal['fp'], t.Literal['first']] = self.click_type(idx)
        return self._error_mask(click_type, mask, prev_mask)
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, val):
        assert val in [None, 'pc', 'mobile']
        self._device = val
    
    def clicks(self, idx, device=None):
        full_stem = self.full_stem(idx)
        df = self.clicks_df[self.clicks_df.full_stem == full_stem]
        
        if device is None:
            device = self._device
        
        if device == None:
            xs, ys = df['x'].to_numpy(), df['y'].to_numpy()
        elif device in ['pc', 'mobile']:
            mask = df.device == device
            xs, ys = df[mask]['x'].to_numpy(), df[mask]['y'].to_numpy()
        else:
            raise RuntimeError(f'Unkown device {device}')
        # assuming that images are loaded in the size of clicks
        # except datasets with 'max_side_size' attribute
        tw, th = df['w'].to_list()[0], df['h'].to_list()[0] # target w and h
        xs, ys = np.clip(xs, 0, tw - 1), np.clip(ys, 0, th - 1)
        max_side_size = None
        if hasattr(self.dataset, 'max_side_size'):
            max_side_size = self.dataset.max_side_size
        if max_side_size is not None:
            w, h = tw, th
            tw, th = _limited_image_size(w, h, max_side_size)
            xs, ys = (xs*tw/w).astype(int), (ys*th/h).astype(int)
        return xs, ys, tw, th

    def sample(self, idx, device=None):
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
        out['xs'], out['ys'], out['w'], out['h'] = self.clicks(idx, device=device)
        return out

    def __getitem__(self, idx):
        return self.sample(idx)   

def display_mode_clicks(display_mode, max_side_size=2048):
    dataset_path = data.PREVIEW_IMG_MASK_DIR
    dataset_obj = data.TETRISDataset(
        dataset_path,
        data._args,
        max_side_size=max_side_size)
    clicks_csv = Path(data.PREVIEWS_CSV)
    prev_mask_dir = 'NO_PREV_MASKS'
    return Clicks(
        display_mode, 
        dataset_obj=dataset_obj,
        clicks_csv=clicks_csv,
        prev_mask_dir=prev_mask_dir
    )

class GaussianMaps(Clicks):

    @classmethod
    def load(
        cls,
        dataset_name,
        files=None,
        calc_func=utils.render_gaze_map,
        **calc_kwargs):
        
        dataset_obj = data.ISEGM_DATASETS[dataset_name]
        clicks_csv = Path(data.CLICKS_CSV)
        prev_mask_dir = Path(data.PREV_MASK_PATHS[dataset_name])
        self = cls(dataset_name, 
                   dataset_obj=dataset_obj,
                   clicks_csv=clicks_csv,
                   prev_mask_dir=prev_mask_dir,
                   files=files,
                   calc_func=calc_func,
                   **calc_kwargs)
        return self

    def __init__(
            self, 
            dataset_name,
            dataset_obj: isegm.data.base.ISDataset,
            clicks_csv: t.Union[Path, str],
            prev_mask_dir: t.Union[Path, str],
            files=None,
            calc_func=utils.render_gaze_map,
            **calc_kwargs
            ):
        
        self.calc_func = calc_func
        self.calc_kwargs = calc_kwargs
        super().__init__(
            dataset_name=dataset_name, 
            dataset_obj=dataset_obj,
            clicks_csv=clicks_csv,
            prev_mask_dir=prev_mask_dir,
            files=files,
        )

    def gaussian_map(self, idx, device=None, **calc_kwargs):
        xs, ys, w, h = self.clicks(idx, device=device)
        kwargs = dict(**self.calc_kwargs)
        kwargs.update(**calc_kwargs)
        gauss_map = self.calc_func(xs, ys, height=h, width=w, **kwargs)
        return gauss_map / gauss_map.max()
    
    def sample(self, idx, device=None, **calc_kwargs):
        out = super().sample(idx, device=device)
        out['gaussian_map'] = self.gaussian_map(idx, device=device, **calc_kwargs)
        return out


class ClickMaps(GaussianMaps):

    def _click_map(self, gauss_map, error_mask):
        click_radius = int(5 * ((error_mask.shape[0]**2 + error_mask.shape[1]**2)**0.5) /
                     (500 * np.sqrt(2)))
        gaussian_error_mask = gaussian_filter(error_mask, sigma=click_radius)
        click_map = gaussian_error_mask * gauss_map
        return click_map

    def click_map(self, idx, device=None, **calc_kwargs):
        kwargs = dict(**self.calc_kwargs)
        kwargs.update(**calc_kwargs)
        gauss_map = self.gaussian_map(idx, device=device, **kwargs)
        error_mask = self.error_mask(idx)
        return self._click_map(gauss_map, error_mask)
    
    def sample(self, idx, device=None, **calc_kwargs):
        out = super().sample(idx, device=device, **calc_kwargs)
        out['click_map'] = self._click_map(out.gaussian_map, out.error_mask)
        return out


class CachedClickMaps(Clicks):

    @classmethod
    def load(
        cls,
        dataset_name,
        cache_dir: Path,
        files=None,
        ):
        
        dataset_obj = data.ISEGM_DATASETS[dataset_name]
        clicks_csv = Path(data.CLICKS_CSV)
        prev_mask_dir = Path(data.PREV_MASK_PATHS[dataset_name])
        self = cls(dataset_name, 
                   dataset_obj=dataset_obj,
                   clicks_csv=clicks_csv,
                   prev_mask_dir=prev_mask_dir,
                   cache_dir=cache_dir,
                   files=files,)
        return self

    def __init__(
            self, 
            dataset_name,
            dataset_obj: isegm.data.base.ISDataset,
            clicks_csv: t.Union[Path, str],
            prev_mask_dir: t.Union[Path, str],
            cache_dir: Path,
            files=None,
            ):
        
        self.cache_dir = Path(cache_dir)
        assert self.cache_dir.exists()
        super().__init__(
            dataset_name=dataset_name, 
            dataset_obj=dataset_obj,
            clicks_csv=clicks_csv,
            prev_mask_dir=prev_mask_dir,
            files=files,
        )

    def _click_map(self, gauss_map, error_mask):
        click_radius = int(5 * ((error_mask.shape[0]**2 + error_mask.shape[1]**2)**0.5) /
                     (500 * np.sqrt(2)))
        gaussian_error_mask = gaussian_filter(error_mask, sigma=click_radius)
        click_map = gaussian_error_mask * gauss_map
        return click_map

    def cached_path(self, idx):
        full_stem = self.full_stem(idx)
        path = self.cache_dir / f'{full_stem}.npy'
        return path

    def click_map(self, idx):
        path = self.cached_path(idx)
        assert path.is_file()
        return np.load(path)
        
    def sample(self, idx, device=None, **calc_kwargs):
        out = super().sample(idx, device=device, **calc_kwargs)
        out['click_map'] = self.click_map(idx)
        return out
