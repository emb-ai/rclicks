
import mmengine.logging.logger
import os

def _get_device_id():
    """Get device id of current machine."""
    try:
        import torch
    except ImportError:
        return 0
    else:
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        world_size = int(os.getenv('WORLD_SIZE', '1'))
        # TODO: return device id of npu and mlu.
        if not torch.cuda.is_available():
            return local_rank
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible_devices is None:
            num_device = torch.cuda.device_count()
            cuda_visible_devices = list(range(num_device))
        else:
            cuda_visible_devices = cuda_visible_devices.split(',')

        if world_size > len(cuda_visible_devices):
            local_rank = local_rank % len(cuda_visible_devices)

        try:
            return int(cuda_visible_devices[local_rank])
        except ValueError:
            # handle case for Multi-Instance GPUs
            # see #1148 for details
            return cuda_visible_devices[local_rank]

mmengine.logging.logger._get_device_id = _get_device_id

import numpy as np
import math
import cv2
from torchvision import transforms, models
from pathlib import Path
import torch
from . import paths

def get_defualt_ksize(sigma):
    # 3.5 * sigma rule
    diameter = int(math.ceil(7 * sigma))
    diameter += (diameter + 1) % 2
    ksize = (diameter, diameter)
    return ksize

def padding(img, height=416, width=416, channels=3):
    channels = img.shape[2] if len(img.shape) > 2 else 1
    
    if channels == 1:
        img_padded = np.zeros((height, width), dtype=img.dtype)
    else:
        img_padded = np.zeros((height, width, channels), dtype=img.dtype)

    original_shape = img.shape
    rows_rate = original_shape[0] / height
    cols_rate = original_shape[1] / width

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * height) // original_shape[0]
        img = cv2.resize(img, (new_cols, height))
        if new_cols > width:
            new_cols = width
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):
                   ((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        
        new_rows = (original_shape[0] * width) // original_shape[1]
        
        img = cv2.resize(img, (width, new_rows))
        if new_rows > height:
            new_rows = height
#         print(((img_padded.shape[0] - new_rows) // 2))
#         print((img_padded.shape[0] - new_rows) // 2 + new_rows)
        img_padded[((img_padded.shape[0] - new_rows) // 2):
                   ((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def unpadding(img_padded, original_shape):
    height, width = img_padded.shape[2:]


    original_shape = original_shape
    rows_rate = original_shape[0] / height
    cols_rate = original_shape[1] / width
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * height) // original_shape[0]
        if new_cols > width:
            new_cols = width
        
        img = img_padded[:, :, :, ((img_padded.shape[3] - new_cols) // 2):
                   ((img_padded.shape[3] - new_cols) // 2 + new_cols)]
    else:
        new_rows = (original_shape[0] * width) // original_shape[1]
        if new_rows > height:
            new_rows = height
        img = img_padded[:, :, ((img_padded.shape[2] - new_rows) // 2):
                   ((img_padded.shape[2] - new_rows) // 2 + new_rows), :]
        
    resize_transform = transforms.Resize((original_shape[0], original_shape[1]))
    img = resize_transform(img)

    return img


def click_distibution(
    clicks_coords,
    height,
    width,
    sigma_=120,
    height_=1080,
    width_=1920,
    ):

    click_map = np.zeros((height, width), dtype=np.float32)
    sigma = sigma_ * np.sqrt(height**2 + width**2) / np.sqrt(height_**2 + width_**2)
    for coords in clicks_coords:
        click_map[coords[1], coords[0]] += 1

    click_map = cv2.GaussianBlur(
        click_map,
        ksize=get_defualt_ksize(sigma), 
        sigmaX=sigma, 
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT)
    
    return click_map

def get_normalizer(sigma):
    return (2 * math.pi * sigma**2)

def render_gaze_map(
    xs, ys, 
    height,
    width,
    sigma=120,
    max_value=255,
    ):
    
    assert xs.shape == ys.shape
    sigma = sigma * np.sqrt(height ** 2 + width ** 2) / np.sqrt(1920 ** 2 + 1080 ** 2)

    sm = np.zeros((height, width), dtype=np.float32)
    num_observers = len(xs)

    # If we have one fixation, its max value will be 255
    max_value = 255 if max_value is None else max_value
    base_weight = max_value * get_normalizer(sigma)
    weight = base_weight / num_observers

    for x, y in zip(xs, ys):
        sm[y, x] += weight

    sm = cv2.GaussianBlur(sm, ksize=get_defualt_ksize(sigma), 
                          sigmaX=sigma, 
                          sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)
    return sm

def get_bbox_from_mask(mask, dialate_size=None):
    
    if dialate_size:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dialate_size,dialate_size))
        # dilate_th = cv2.dilate(th, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax