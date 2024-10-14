import torch
from mmengine.registry import MODELS
from mmseg.registry import MODELS as MODELS_SEG
from mmseg.models.backbones.mscan import MSCAN, StemConv
from mmseg.models.segmentors import BaseSegmentor, EncoderDecoder
from mmseg.models.segmentors import CascadeEncoderDecoder
from mmseg.models.decode_heads import DepthwiseSeparableASPPHead
from mmseg.models.utils import resize