from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker
from joblib import Parallel, delayed
from rclicks.nets import SegNeXtSaliencyApply
import json
import pandas as pd
from rclicks.paths import CLICKS_CSV
import os
import random


def setup_deterministic(seed):
    """
    set every seed
    """
    torch.set_printoptions(precision=16)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_fixations(path):
    clicks_df = pd.read_csv(path, keep_default_na=False)
    clicks_df = clicks_df[clicks_df['click_type'] == 'first']

    parsed_jsons = {}
    
    for dataset_name in set(clicks_df['dataset']):
        df_dataset = clicks_df[clicks_df['dataset'] == dataset_name]
        parsed_jsons[dataset_name] = {}
        
        for full_stem in set(df_dataset['full_stem']):
            parsed_jsons[dataset_name][full_stem] = np.array(df_dataset[df_dataset['full_stem'] == full_stem][['x', 'y']])

    return parsed_jsons

FIRST_ROUND_JSON = parse_fixations(CLICKS_CSV)


try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_functor(dataset, predictor, index, click_model, **kwargs):

    setup_deterministic(kwargs['args'].seed)
    
    all_ious = []
    
    sample = dataset.get_sample(index)
    
    for obj_id in sample.objects_ids:
        
        gt_mask = sample.get_object_mask(obj_id)
        
        user_inputs = None
        
        dataset_name = dataset.__class__.__name__
        if 'GrabCut' in dataset_name:
            dataset_name = 'GrabCut'  
        elif 'Berkeley' in dataset_name:
            dataset_name = 'Berkeley'
        elif 'Davis' in dataset_name and 'COCO' not in str(dataset.dataset_path):
            dataset_name = 'DAVIS'
        elif 'Davis' in dataset_name:
            dataset_name = 'COCO'
        elif 'TETRIS' in dataset_name:
            dataset_name = 'TETRIS'
        else:
            raise "No user inputs for such dataset"

        layer_indx, mask_id = sample._objects[obj_id]['mapping']
        
        if kwargs['args'].user_inputs:
            sample_name = sample.imname.split('/')[-1][:-4]
            if len(sample.objects_ids) > 1:
                sample_name = sample_name + "_______" + str(mask_id)
            if sample_name in FIRST_ROUND_JSON[dataset_name]:
                user_inputs = FIRST_ROUND_JSON[dataset_name][sample_name]
            else:
                continue

        _, sample_ious, _ = evaluate_sample(sample.image, gt_mask, predictor, click_model,
                                            sample_id=index, user_inputs=user_inputs, dataset_name=dataset_name, **kwargs)
        all_ious.append([sample.imname, sample_ious, mask_id])

    return all_ious


def evaluate_dataset(dataset, predictor, **kwargs):
    
    all_ious = []
    start_time = time()
    dataset_iterator = tqdm(range(len(dataset)), leave=False)
    
    if kwargs['args'].clickability_model_pth is not None:
        #SegNext model
        pth_path = kwargs['args'].clickability_model_pth
        click_model = SegNeXtSaliencyApply(pth_path).to('cuda') 
        print("Using clickablity model", pth_path)
        click_model.eval()
    else:
        print("Not use clickability model")
        click_model = None


    if kwargs['args'].n_workers == 1:
        for index in dataset_iterator:
            all_ious.append(evaluate_functor(dataset, predictor, index, click_model, **kwargs))
    else:
        all_ious = Parallel(n_jobs=kwargs['args'].n_workers)(delayed(evaluate_functor)(dataset, predictor, index, click_model, **kwargs) for index in dataset_iterator)
    
    # Flatten list of lists
    all_ious = [item for sublist in all_ious for item in sublist]
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time



def evaluate_sample(image, gt_mask, predictor, click_model, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, args=None, user_inputs=None, dataset_name=None):

    if user_inputs is not None:

        ious_list = []
        clicks_list = []
        
        for idx, user_click in enumerate(user_inputs):

            # if dataset_name is not None and 'TETRIS' in dataset_name:
            #     # Since we downscaled TETRIS dataset to max 2048p
            #     # in the clicks crowdsoursing collection phase
            #     scaling = np.max(gt_mask.shape / np.max(gt_mask.shape)
            #     user_click[0] = user_click[0] / scaling
            #     user_click[1] = user_click[1] / scaling
            #     print(scaling)
                
            if click_model is not None:
                clicker = Clicker(gt_mask=gt_mask, image=image, click_model=click_model)
            else:
                clicker = Clicker(image=image)

            pred_mask = np.zeros_like(gt_mask)
            predictor.set_input_image(image)

            with torch.no_grad():
                for click_indx in range(max_clicks):
                    
                    if click_model is not None:
                        clickmap, click = clicker.make_next_click(pred_mask)
                        user_click = click.coords
                    else:
                        clickmap = clicker.make_next_click_coords(user_click)

                    pred_probs = predictor.get_prediction(clicker)
                    pred_mask = pred_probs > pred_thr
                                       
                    if clickmap is None:
                        clickmap = pred_probs

                    if callback is not None:
                        callback(image, gt_mask, pred_probs, sample_id, click_indx, 
                                 clicker.clicks_list, clickmap)
                    iou = utils.get_iou(gt_mask, pred_mask)
                    biou = utils.get_boundary_iou(gt_mask, pred_mask)
                    ious_list.append([iou, biou, user_click])
                    if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                        break

    elif click_model is not None:

        with torch.no_grad():
            ious_list = []
            predictor.set_input_image(image)
            pred_mask = np.zeros_like(gt_mask)
            clicker = Clicker(gt_mask=gt_mask, image=image, click_model=click_model, 
                              quantile_low=args.trajectory_sampling_prob_low, 
                              quantile_high=args.trajectory_sampling_prob_high)

            for click_indx in range(max_clicks):
                
                clickmap, click = clicker.make_next_click(pred_mask)
                pred_probs = predictor.get_prediction(clicker)
                pred_mask = pred_probs > pred_thr

                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, 
                             click_indx, clicker.clicks_list, clickmap)

                iou = utils.get_iou(gt_mask, pred_mask)
                biou = utils.get_boundary_iou(gt_mask, pred_mask)

                ious_list.append([iou, biou, click])
                
                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    break

    # Baseline
    else:
        
        clicker = Clicker(gt_mask=gt_mask, image=image)
        pred_mask = np.zeros_like(gt_mask)
        ious_list = []

        predictor.set_input_image(image)

        with torch.no_grad():

            for click_indx in range(max_clicks):

                clickmap, click = clicker.make_next_click(pred_mask)

                pred_probs = predictor.get_prediction(clicker)

                pred_mask = pred_probs > pred_thr

                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list, clickmap)

                iou = utils.get_iou(gt_mask, pred_mask)
                biou = utils.get_boundary_iou(gt_mask, pred_mask)

                ious_list.append([iou, biou, click])

                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    break

    
    return clicker.clicks_list, np.array(ious_list), pred_probs
