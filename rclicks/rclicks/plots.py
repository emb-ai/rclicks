import cv2
import matplotlib.pyplot as plt
import numpy as np
import functools
from easydict import EasyDict as edict
import typing as t

def colorize(img, cv2_color_map=cv2.COLORMAP_JET, **kwargs):
    """
    Colorize grey scale maps
    """
    assert len(img.shape) == 2
    if cv2_color_map is None:
        return np.dstack([img, img, img])
    img = cv2.applyColorMap(img, cv2_color_map)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def image_resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=interpolation)
    return resized


def preproc(img: np.ndarray, cv2_color_map=cv2.COLORMAP_JET, gamma=None, eps=10e-7, **kwargs):
    """
    Preprocess float image before plotting
    """
    assert img.dtype != np.uint8
    if gamma is not None:
        img = img**gamma
    img = (img / (img.max() + eps) * 255).astype(np.uint8)
    if len(img.shape) == 3:
        return img
    else:
        return colorize(img, cv2_color_map)

def mix(img1: np.ndarray, img2: np.ndarray, q=0.5, **kwargs):
    """
    Linearly combine two images
    """
    return (img1*q + img2*(1-q)).astype(np.uint8)

def clicks_over_image(xs, ys, img: np.ndarray, radius_ratio=1., 
                      color=(0, 255, 0),
                      boundary_color=(255, 0, 0),
                      **kwargs):
    """
    Put clicks over the image
    """
    assert img.dtype == np.uint8
    diag = (img.shape[0]**2 + img.shape[1]**2)**0.5
    radius = round((7. / 400.) * diag * radius_ratio)
    small_radius = round((6. / 400.) * diag * radius_ratio)
    
    for cX, cY in zip(xs, ys):
        img = cv2.circle(img, (cX, cY), radius, boundary_color, -1)
        img = cv2.circle(img, (cX, cY), small_radius, color, -1)

    return img

def zoomin(img, bbox):
    rmin, rmax, cmin, cmax = bbox
    return img[rmin: rmax, cmin: cmax]

def mask_boundary_over_image(img, mask, radius_ratio, color=(255, 255, 255)):
    """
    Plot countour of the mask's boundary over the image
    """
    diag = (img.shape[0]**2 + img.shape[1]**2)**0.5
    radius = round((7. / 400.) * diag * radius_ratio)
    small_radius = round((6. / 400.) * diag * radius_ratio)

    contours = cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
    res = (img*255).astype(np.uint8)
    cv2.drawContours(res, contours, -1, color, radius)
    # cv2.drawContours(res, contours, -1, (0, 255, 0), small_radius)
    return res

def plotter_init(**figure_kwargs):
    """
    Initialize some standard parameters for matplotlib.pytplot plotter 
    """
    plt.figure(**figure_kwargs)
    plt.axis('off')

def from_sample_plotter(sample: edict, key, call_imshow=True, **kwargs):
    """
    Plot the data correspnding to the key from rclicks.ClickMap sample
    """
    img = preproc(sample[key], **kwargs)
    if call_imshow:
        plt.imshow(img)
    return img
    
def on_image_plotter(sample: edict, key: str=None, vis_clicks=False, call_imshow=True, q_mix=0.35, **kwargs):
    """
    Plot the data correspnding to the key from rclicks.ClickMaps sample
    with clicks over the image from sample 
    """
    img_orig = preproc(sample['image'])
    if key is None:
        img = img_orig
    else:
        img_mask = preproc(sample[key], **kwargs)
        img = mix(img_orig,  img_mask, q_mix)
    if vis_clicks:
        img = clicks_over_image(sample.xs, sample.ys, img, **kwargs)
    if call_imshow:
        plt.imshow(img)
    return img

def sample_plotter_impl(sample: edict, visulization_funcs: t.Dict[str, t.Callable], subplot_shape_int: int, **kwargs):
    """
    Given a set of visualizaion functions for some of keys
    or thier mix, plot rclicks.ClickMaps sample
    """
    plt.suptitle(f"{sample['dataset']} {sample['full_stem']}")
    for i, (title, vis_func) in enumerate(visulization_funcs.items()):
        plt.subplot(subplot_shape_int + i)
        plt.title(title)
        vis = vis_func(sample)
        plt.imshow(vis)
        plt.axis('off')

def clicks_plotter(sample: edict, **kwargs):
    """
    Plot image with clicks, mask, and error mask from rclicks.Clicks sample
    """
    visulization_funcs = dict(
        image=functools.partial(on_image_plotter, key=None, vis_clicks=True, call_imshow=False), 
        error_mask=functools.partial(from_sample_plotter, key='error_mask', call_imshow=False),
        mask=functools.partial(from_sample_plotter, key='mask', call_imshow=False),
    )
    sample_plotter_impl(sample, visulization_funcs, 131, **kwargs)

def clickmaps_minimal_plotter(sample: edict, **kwargs):
    """
    Plot image with clicks, error mask and click map from rclicks.ClickMaps sample
    """
    visulization_funcs = dict(
        image=functools.partial(on_image_plotter, key=None, vis_clicks=True, call_imshow=False), 
        error_mask=functools.partial(from_sample_plotter, key='error_mask', call_imshow=False),
        click_map=functools.partial(from_sample_plotter, key='click_map', show_clicks=False),
    )
    sample_plotter_impl(sample, visulization_funcs, 131, **kwargs)

def clickmaps_full_plotter(sample, **kwargs):
    """
    Plot all keys from rclicks.ClickMaps sample
    """
    visulization_funcs = dict(
        image=functools.partial(on_image_plotter, key=None, vis_clicks=True, call_imshow=False), 
        mask=functools.partial(from_sample_plotter, key='mask', call_imshow=False),
        error_mask=functools.partial(from_sample_plotter, key='error_mask', call_imshow=False),
        gaussian_map=functools.partial(from_sample_plotter, key='gaussian_map', show_clicks=False),
        click_map=functools.partial(from_sample_plotter, key='click_map', show_clicks=False), 
    )
    sample_plotter_impl(sample, visulization_funcs, 231, **kwargs)
