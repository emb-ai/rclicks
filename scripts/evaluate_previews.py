import rclicks.ndtest as ndtest
import scipy
import numpy as np
from easydict import EasyDict as edict
from rclicks import ClickMaps, display_mode_clicks
from rclicks import data, paths
import typing as t
import tqdm.auto as tqdm
import itertools
import pandas as pd


def calculate_distance(x1, y1, x2, y2, height, width):
    return abs(x2 - x1) / height + abs(y2 - y1) / width

def get_height_width(mask):
    # Find the indices of 1s in the mask
    rows, cols = np.where(mask == 1)
    
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0
    
    # Calculate the height and width
    height = np.max(rows) - np.min(rows) + 1
    width = np.max(cols) - np.min(cols) + 1
    
    return height, width

def l1(clicks, gt_clicks, size=None, eps=1e-9, **kwargs):
    w, h = size
    xs_gt, ys_gt = gt_clicks[:, 0], gt_clicks[:, 1]
    xs, ys = clicks[:, 0], clicks[:, 1]

    m = np.c_[xs / w, ys / h]
    g = np.c_[xs_gt / w, ys_gt / h]
    pairs = np.array(list(itertools.product(g, m)))
    results = np.linalg.norm(pairs[:, 0, :] - pairs[:, 1, :], axis=1, ord=1)
    return np.mean(results)

def ks2d(clicks, gt_clicks, eps=1e-9, **kwarg):
    xs_gt, ys_gt = gt_clicks[:, 0], gt_clicks[:, 1]
    xs, ys = clicks[:, 0], clicks[:, 1]
    results = []
    prob_ks = ndtest.ks2d2s(ys_gt, xs_gt, ys, xs)
    results.append(prob_ks > 0.05)
    return np.mean(results)

def wasserstien(clicks, gt_clicks, size, **kwargs):
    w, h = size
    return scipy.stats.wasserstein_distance_nd(clicks, gt_clicks) / np.sqrt(h**2 + w**2)
    
def calculate_metric(ideal_dm, dm, metrics: t.Dict[str, t.Callable]):
    metric_result = {k: [] for k in metrics.keys()}
    for full_stem in tqdm.tqdm(ideal_dm.full_stems, leave=False):
        ideal_idx, idx = ideal_dm.indx(full_stem), dm.indx(full_stem)
        ideal_clicks, clicks = ideal_dm.clicks(ideal_idx), dm.clicks(idx)
        
        ideal_xs, ideal_ys, ideal_w, ideal_h = ideal_clicks
        xs, ys, w, h = clicks
        assert ideal_w == w
        assert ideal_h == h
        
        mask = ideal_dm.mask(ideal_idx)
        h, w = get_height_width(mask)
        size = (w, h)

        ideal_clicks = np.dstack([ideal_xs, ideal_ys])[0]
        clicks = np.dstack([xs, ys])[0]

        for metric_name, metric in metrics.items():
            metric_result[metric_name].append(metric(clicks, ideal_clicks, size))
    return metric_result

if __name__=='__main__':
    display_modes = edict(
        **{
            d: display_mode_clicks(d, 2048) for d in data.display_modes
        },
    )
    # display_modes_names = ['Text_Description', 'Object_CutOut', 'Shifted_CutOut', 'Silhouette_Mask', 'Highlighted_Instance']

    def return_metrics_for_all(metrics: t.Dict[str, t.Callable]):
        ideal_dm = display_modes.Text_Description
        
        results = []
        for name, dm  in display_modes.items():
            if name == 'Text_Description':
                continue
            dict_of_metrics = calculate_metric(ideal_dm, dm, metrics)
            dict_of_metrics = {key: np.mean(value) for key, value in dict_of_metrics.items()}
            results.append(dict(
                name=name,
                **dict_of_metrics))
            
        df = pd.DataFrame(results)
        return df

    metrics = dict(
        l1=l1, 
        ks2d=ks2d, 
        wasserstien=wasserstien,
    )

    df = return_metrics_for_all(metrics)
    print(df)
