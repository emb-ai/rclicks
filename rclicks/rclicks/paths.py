import sys
from pathlib import Path

PKG_DIR = Path(__file__).absolute().parent.parent
BENCHMARK_DIR = PKG_DIR.parent

from easydict import EasyDict as edict

DATASETS_DIR = BENCHMARK_DIR / Path('datasets')

CLICKS_CSV = DATASETS_DIR / 'clicks.csv'

PREVIEW_IMG_MASK_DIR = DATASETS_DIR / 'PREVIEW'

PREVIEWS_CSV = DATASETS_DIR / 'previews.csv'

ISEGM_DATASETS_DIRS = edict({
    'TETRIS': DATASETS_DIR / 'TETRIS',
    'DAVIS': DATASETS_DIR / 'DAVIS',
    'GrabCut': DATASETS_DIR / 'GrabCut',
    'Berkeley': DATASETS_DIR / 'Berkeley',
    'COCO': DATASETS_DIR / 'COCO_MVal',
})

PREV_MASK_PATHS = {
    'TETRIS': DATASETS_DIR / 'SUBSEC_MASKS/tetris_fn_fp/tetris_fn_fp_masks',
    'DAVIS': DATASETS_DIR / 'SUBSEC_MASKS/all_benchmark_datasets/fn_fp',
    'GrabCut': DATASETS_DIR / 'SUBSEC_MASKS/all_benchmark_datasets/fn_fp',
    'Berkeley': DATASETS_DIR / 'SUBSEC_MASKS/all_benchmark_datasets/fn_fp',
    'COCO': DATASETS_DIR / 'SUBSEC_MASKS/all_benchmark_datasets/fn_fp',
}

TETRIS_TRAIN = DATASETS_DIR / 'TETRIS_train.txt'
TETRIS_VAL = DATASETS_DIR / 'TETRIS_val.txt'

MODELS_DIR = BENCHMARK_DIR
MODEL_FNAME = 'clickability_model.pth'
MODELS_FNAME = 'clickability_model_{sigma}.pth'