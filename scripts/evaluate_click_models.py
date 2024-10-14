import click
import scipy
import pandas as pd
import numpy as np
import typing as t
import tqdm.auto as tqdm
import rclicks
from rclicks import data, paths
import rclicks.ndtest as ndtest
from pathlib import Path
import cv2

import torch
import os
import itertools


def calculate_distance(x1, y1, x2, y2, w, h):
    return abs(x2 - x1) / w + abs(y2 - y1) / h

def get_height_width(mask):
    rows, cols = np.where(mask == 1)
    
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0
    height = np.max(rows) - np.min(rows) + 1
    width = np.max(cols) - np.min(cols) + 1
    
    return height, width

def sample_from_dist(prob_map, num_points=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # np.random.seed(42)
    prob_map = prob_map / prob_map.sum()
    flat = prob_map.flatten()
    sample_index = np.random.choice(a=flat.size, p=flat, size=num_points)
    adjusted_index = np.unravel_index(sample_index, prob_map.shape)
    xs, ys, w, h = adjusted_index[1], adjusted_index[0], prob_map.shape[1], prob_map.shape[0] 
    return xs, ys, w, h

BNUM = 100

def wasserstien(s_map, x_gt, y_gt, size):
    s_map = s_map / np.sum(s_map)
    width, height = size
    gt_clicks = np.dstack([y_gt, x_gt])[0]
    x_s_map_list, y_s_map_list, _, _ = sample_from_dist(s_map,
                                                num_points=len(x_gt)*BNUM)
    x_s_map_list, y_s_map_list = np.reshape(
        x_s_map_list, (BNUM, len(x_gt))), np.reshape(
        y_s_map_list, (BNUM, len(x_gt)))
    
    results = []
    for i in range(BNUM):
        x, y, = x_s_map_list[i], y_s_map_list[i]
        
        was = scipy.stats.wasserstein_distance_nd(
            np.dstack([y, x])[0],
            gt_clicks,
            ) / np.sqrt(height**2 + width**2)
        results.append(was)
    return np.mean(results)

def mean_pde_by_click(s_map, x_gt, y_gt, eps=1e-9, **kwargs):
    s_map = s_map / (s_map.sum() + eps)
    temp = s_map[y_gt, x_gt]
    return np.mean(temp)

def l1(s_map, x_gt, y_gt, size, **kwargs):
    width, height = size
    s_map = s_map / np.sum(s_map)
    
    x_s_map_list, y_s_map_list, _, _ = sample_from_dist(s_map,
                                                num_points=len(x_gt)*BNUM)
    x_s_map_list, y_s_map_list = np.reshape(
        x_s_map_list, (BNUM, len(x_gt))) / width, np.reshape(
        y_s_map_list, (BNUM, len(x_gt))) / height
        
    x_gt, y_gt = x_gt / width, y_gt / height
    
    gt = np.c_[x_gt, y_gt]
    
    results = []
    for i in range(BNUM):
        x_s_map, y_s_map = x_s_map_list[i], y_s_map_list[i]
        m = np.c_[x_s_map, y_s_map]
        pairs = np.array(list(itertools.product(gt, m)))
        temp = np.linalg.norm(pairs[:, 0, :] - pairs[:, 1, :], axis=1, ord=1)
        results.append(temp)
    return np.mean(results)

def nss(s_map, x_gt, y_gt, eps=1e-9, **kwargs):
    s_map = s_map / (s_map.sum() + eps)
    s_map_norm = (s_map - np.mean(s_map)) / (np.std(s_map) + eps)
    x_gt = np.clip(x_gt, 0, s_map.shape[1] - 1)
    y_gt = np.clip(y_gt, 0, s_map.shape[0] - 1)
    temp = s_map_norm[y_gt, x_gt]
    return np.mean(temp)

def ks2d(s_map, x_gt, y_gt, **kwargs):
    # np.random.seed(42)
    s_map = s_map / np.sum(s_map)
    
    x_s_map_list, y_s_map_list, _, _ = sample_from_dist(s_map,
                                                num_points=len(x_gt)*BNUM)
    x_s_map_list, y_s_map_list = np.reshape(
        x_s_map_list, (BNUM, len(x_gt))), np.reshape(
        y_s_map_list, (BNUM, len(x_gt)))
    
    results = []
    for i in range(BNUM):
        x_s_map, y_s_map, = x_s_map_list[i], y_s_map_list[i]
        prob_ks = ndtest.ks2d2s(x_gt, y_gt, x_s_map, y_s_map)
        results.append(prob_ks > 0.05)
    return np.mean(results)


RANK = int(os.environ.get("RANK", '0'))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", '0'))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", '1'))
CUDA_VISIBLE_DEVICES = [int(d) for d in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
GPUS_SIZE = len(CUDA_VISIBLE_DEVICES)

def ddp_setup():
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    if WORLD_SIZE > GPUS_SIZE:
        gpu_num = RANK % GPUS_SIZE
    else:
        gpu_num = RANK
    torch.cuda.set_device(gpu_num)
    torch.distributed.init_process_group(backend='gloo', rank=RANK, world_size=WORLD_SIZE)
  


@click.command('Compare clickability model with baselines models')
@click.option('-o', '--outdir', type=Path,
              default=Path('experiments'),
              help='Directory to save results'
)
@click.option('-d', '--dataset', type=str,
              default='TETRIS',
              help='Name of dataset for evaluation'
)
@click.option('-c', '--cv2_threads', type=int,
              default=-1,
              help='Number of threads for `cv2.setNumThreads`'
)
@click.option('-s', '--sigma_ablations_dir', type=Path,
              default=None,
              help='Path to directory with sigma ablation models'
)
def main(outdir, dataset, cv2_threads, sigma_ablations_dir):
    outdir.mkdir(parents=True, exist_ok=True)

    if dataset == 'TETRIS':
        files = data.TETRIS_VAL
    else:
        files = None
    
    gt = rclicks.Clicks.load(dataset_name=dataset, files=files)
    sim = rclicks.SimClicks.load(dataset_name=dataset, files=files, 
                                model=None,
                                seed=42)
    if WORLD_SIZE > 1:
        ddp_setup()
    
    is_master = RANK == 0
    
    
    models = dict(
        ud=rclicks.uniform_model,
        dt=rclicks.distance_transform_model,
        si=rclicks.SaliencyImageModel(rclicks.load_transalnet_model()),
        sm=rclicks.SaliencyModel(rclicks.load_transalnet_model()),
        cm=rclicks.ClickabilityModel(rclicks.load_segnext(
            paths.MODEL_FNAME)),
    )
    
    if sigma_ablations_dir is not None:
        cm_list = [
            '001',
            '025',
            '05',
            '1',
            '2',
            '10',
            '20',
            '30',
            '60',
            '90',
            '120',
        ]
        
        def load_segnext(
            path,
            device='cuda'):
            model = rclicks.SegNeXtSaliency().cuda()
            
            model(torch.randn(1, 3, 100, 100).cuda(), torch.randn(1, 2, 100, 100).cuda())
            model.load_state_dict(torch.load(path))
            model.eval();
            return model
        
        models.update(
            **{
                f'cm_{cm_sigma}':  rclicks.ClickabilityModel(load_segnext(
                    sigma_ablations_dir / paths.MODELS_FNAME.format(sigma=cm_sigma)))
                for cm_sigma in cm_list
            }
        )

    if cv2_threads >= 0:
        cv2.setNumThreads(cv2_threads)

    def job(sim, model_name, i):
        assert gt.full_stem(i) == sim.full_stem(i)
                
        row = dict(full_stem = gt.full_stem(i))
        row['model_name'] = model_name
        row['click_type'] = gt.click_type(i)
        
        xs, ys, w, h = gt.clicks(i)
        click_map = sim.click_map(i)
        
        mask = sim.mask(i)
        assert click_map.shape == mask.shape
        # assert w == click_map.shape[1], f'{w} != {click_map.shape[1]}'
        # assert h == click_map.shape[0], f'{h} != {click_map.shape[0]}'
        h, w = get_height_width(mask)

        for metric_name, metric in metrics.items():
            mval = metric(
                s_map=click_map, 
                x_gt=xs, 
                y_gt=ys,
                size=(w, h))
            row[metric_name] = mval
        return row
    

    def return_metrics_for_all(metrics: t.Dict[str, t.Callable]):
        res = []

        for model_name, model  in models.items():
            sim.model = model
            samples_num = len(gt)
            indexes_all =  list(range(samples_num))
            
            indexes = np.array_split(indexes_all, WORLD_SIZE)[RANK]
            if is_master:
                indexes = tqdm.tqdm(indexes)
            tmp_res = [job(sim, model_name, i) for i in indexes]
            
            if WORLD_SIZE > 1:
                gather_res = [None]*WORLD_SIZE

                torch.distributed.barrier()
                torch.distributed.all_gather_object(gather_res, tmp_res)
                for r in gather_res:
                    res.extend(r)
            else:
                res = tmp_res
            df = pd.DataFrame.from_dict(res)
            if is_master:
                df.to_csv(str(outdir / f'{dataset}_per_image.csv'), index=False)
                df_mean = df.groupby('model_name')[metrics_names].mean()
                df_mean.to_csv(str(outdir / f'{dataset}_mean.csv'))
                print(df_mean)
        
        return df
        
        # if is_master:
        #     df = pd.read_csv(str(outdir / f'{dataset}_per_image.csv'), 
        #                      index_col=False)
        #     df_mean = df.groupby(['model_name', 'click_type'])[metrics_names].mean()
        #     df_mean.to_csv(str(outdir / f'{dataset}_mean.csv'))
        #     print(df_mean)
        #     return df

    np.random.seed(42)

    metrics = dict(
        l1=l1, 
        ks2d=ks2d, 
        wasserstien=wasserstien,
        nss=nss,
        pde=mean_pde_by_click,
    )
    metrics_names = list(metrics.keys())

    df = return_metrics_for_all(metrics)

if __name__=='__main__':
    with torch.no_grad():
        main()
