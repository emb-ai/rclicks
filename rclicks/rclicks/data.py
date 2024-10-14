from .paths import *
import json
from easydict import EasyDict as edict
from isegm.data.datasets import *
from types import SimpleNamespace
import pandas as pd

def clicks_df() -> pd.DataFrame:
    return pd.read_csv(CLICKS_CSV, keep_default_na=False)

def previews_df() -> pd.DataFrame:
    return pd.read_csv(PREVIEWS_CSV, keep_default_na=False)

_args = SimpleNamespace(n_samples=None)

ISEGM_DATASETS = edict({
    'TETRIS': TETRISDataset(ISEGM_DATASETS_DIRS.TETRIS, _args, 
                            max_side_size=2048),
    'DAVIS': DavisDataset(ISEGM_DATASETS_DIRS.DAVIS, _args),
    'GrabCut': GrabCutDataset(ISEGM_DATASETS_DIRS.GrabCut, _args),
    'Berkeley': BerkeleyDataset(ISEGM_DATASETS_DIRS.Berkeley, _args),
    'COCO': DavisDataset(ISEGM_DATASETS_DIRS.COCO, _args)
})


display_modes = (
    'Text_Description',
    'Object_CutOut',
    'Shifted_CutOut',
    'Silhouette_Mask',
    'Highlighted_Instance',
)
