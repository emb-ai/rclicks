import rclicks
from easydict import EasyDict as edict
from tqdm.auto import tqdm
from rclicks.ndtest import ks2d2s
import scipy
import numpy as np
import pandas as pd
from pathlib import Path
import click
import itertools

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

def l1_argmax(clicks, gt_clicks, size=None, eps=1e-9, **kwargs):
    w, h = size
    xs_gt, ys_gt = gt_clicks[:, 0], gt_clicks[:, 1]
    xs, ys = clicks[:, 0], clicks[:, 1]

    m = np.c_[xs / w, ys / h]
    g = np.c_[xs_gt / w, ys_gt / h]
    pairs = np.array(list(itertools.product(g, m)))
    results = np.linalg.norm(pairs[:, 0, :] - pairs[:, 1, :], axis=1, ord=1)
    return np.mean(results)

def wasserstien(clicks, gt_clicks, size, **kwargs):
    w, h = size
    return scipy.stats.wasserstein_distance_nd(clicks, gt_clicks) / np.sqrt(h**2 + w**2)

SIGMA = 5


@click.command('Compare mobile and PC clicks')
@click.option('-o', '--outdir', type=Path,
              default=Path('experiments'),
              help='Directory to save results')
def main(outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    gts = edict(
        GrabCut=rclicks.ClickMaps.load(dataset_name='GrabCut', sigma=SIGMA),
        Berkeley=rclicks.ClickMaps.load(dataset_name='Berkeley', sigma=SIGMA),
        DAVIS=rclicks.ClickMaps.load(dataset_name='DAVIS', sigma=SIGMA),
        COCO=rclicks.ClickMaps.load(dataset_name='COCO', sigma=SIGMA),
        TETRIS=rclicks.ClickMaps.load(dataset_name='TETRIS', files=rclicks.TETRIS_VAL, sigma=SIGMA),
    )

    datasets_names = list(gts.keys())

    res = []

    def job(gt, i, min_num=10):
        full_stem = gt.full_stem(i)
        click_type = gt.click_type(i)

        mask = gt.mask(i)
        h, w = get_height_width(mask)
        size = (w, h)

        pc_xs, pc_ys, pc_w, pc_h = gt.clicks(i, device='pc')
        m_xs, m_ys, m_w, m_h = gt.clicks(i, device='mobile')
        assert pc_w == m_w
        assert pc_h == m_h

        pc_clicks = np.dstack([pc_xs, pc_xs])[0]
        m_clicks = np.dstack([m_xs, m_ys])[0]

        if len(pc_xs) >= min_num and len(m_xs) >= min_num:
            prob_ks = ks2d2s(
                pc_xs, pc_ys, m_xs, m_ys)
            ks_test = float(prob_ks > 0.05)
            w = wasserstien(pc_clicks, m_clicks, size=size)
            l1 = l1_argmax(pc_clicks, m_clicks, size=size)
        else:
            ks_test = np.nan
            prob_ks = np.nan
            w = np.nan
            l1 = np.nan

        return dict(
            full_stem=full_stem,
            dataset=dataset,
            click_type=click_type,
            prob_ks=prob_ks,
            ks_test=ks_test,
            w=w,
            l1=l1,
        )

    for dataset in datasets_names:
        print(f'Evaluating on {dataset}')
        gt = gts[dataset]
        res_tmp = [job(
                gt, i) for i in tqdm(range(len(gt)))
        ]
        res.extend(res_tmp)

    df = pd.DataFrame.from_dict(res)
    df_res = df.groupby('dataset')[['ks_test', 'w', 'l1']].mean()
    print(df_res)
    df_res.to_csv(str(outdir / 'pc_vs_mobile.csv'))


if __name__ == '__main__':
    main()