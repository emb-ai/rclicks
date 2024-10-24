import sys
import pickle
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.utils.exp import load_config_file
from isegm.utils.vis import draw_probmap, draw_with_blend_and_clicks
from isegm.inference.evaluation import evaluate_dataset
import os
import random
from isegm.inference.transforms.zoom_in_gpcis import ZoomIn
from isegm.inference.predictors.gpcis import BaselinePredictor
from evaluate_model_ritm import get_checkpoints_list_and_logs_path, save_results, save_iou_analysis_data, get_prediction_vis_callback
from evaluate_model_ritm import get_predictor_and_zoomin_params, parse_args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', choices=[ 'CDNet', 'Baseline', 'FocalClick', 'NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                         'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
                        help='')

    group_checkpoints = parser.add_mutually_exclusive_group(required=True)
    group_checkpoints.add_argument('--checkpoint', type=str, default='',
                                   help='The path to the checkpoint. '
                                        'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                                        'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--model_dir', type=str, default='',
                                   help='The path to the checkpoint.')

    group_checkpoints.add_argument('--exp-path', type=str, default='',
                                   help='The relative path to the experiment with checkpoints.'
                                        '(relative to cfg.EXPS_PATH)')

    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,SBD,PascalVOC',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, SBD, PascalVOC')

    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0',
                              help='ID of used GPU.')
    group_device.add_argument('--cpu', action='store_true', default=False,
                              help='Use only CPU for inference.')

    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument('--target-iou', type=float, default=0.90,
                                  help='Target IoU threshold for the NoC metric. (min possible value = 0.8)')
    group_iou_thresh.add_argument('--iou-analysis', action='store_true', default=False,
                                  help='Plot mIoU(number of clicks) with target_iou=1.0.')

    parser.add_argument('--n-clicks', type=int, default=20,
                        help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--min-n-clicks', type=int, default=1,
                        help='Minimum number of clicks for the evaluation.')
    parser.add_argument('--thresh', type=float, required=False, default=0.49,
                        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--clicks-limit', type=int, default=None)
    parser.add_argument('--eval-mode', type=str, default='cvpr',
                        help='Possible choices: cvpr, fixed<number> (e.g. fixed400, fixed600).')

    parser.add_argument('--save-ious', action='store_true', default=False)
    parser.add_argument('--print-ious', action='store_true', default=False)
    parser.add_argument('--vis-preds', action='store_true', default=False)
    group_checkpoints.add_argument('--vis_path', type=str, default='./experiments/vis_val/',
                                   help='saveing path for the evaluation results')
    parser.add_argument('--model-name', type=str, default=None,
                        help='The model name that is used for making plots.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--logs-path', type=str, default='',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')
    parser.add_argument('--infer-size', type=int, default=384,
                        help='Inference input size for the model')

    parser.add_argument('--target-crop-r', type=float, default=1.40,
                                  help='Target Crop Expand Ratio')
    
    parser.add_argument('--focus-crop-r', type=float, default=1.40,
                                  help='Focus Crop Expand Ratio')
    
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers on inference')
    parser.add_argument('--minimize', action='store_true', default=False, help='Minimization of iou during optimization')
    parser.add_argument('--n_samples', type=int, default=0, help='Slice only N samples from dataset (for debug only)')
    parser.add_argument('--clickability_model_pth', type=str, default=None, help='Path to clickability model')
    parser.add_argument('--user_inputs', action='store_true', default=False, help='Use user inputs mode (if clickability_model_pth specified, we sample exact number of clicks as users, otherwise use real-users clicks)')
    parser.add_argument('--seed', type=int, default=42, help='Set seed for sampling, keep default for reproducibility')
    parser.add_argument('--trajectory_sampling_prob_low', type=float, default=0.0, help='Sampling from clickmap with prob >=')
    parser.add_argument('--trajectory_sampling_prob_high', type=float, default=1.0, help='Sampling from clickmap with prob <=')

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH / 'evaluation_logs'
    else:
        args.logs_path = Path(args.logs_path)

    return args, cfg




def get_predictor(net, brs_mode, device,
                  prob_thresh=0.49,
                  infer_size = 256,
                  focus_crop_r= 1.4,
                  with_flip=False,
                  zoom_in_params=dict(),
                  predictor_params=None,
                  brs_opt_func_params=None,
                  lbfgs_params=None):

    predictor_params_ = {
        'optimize_after_n_clicks': 1
    }

    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    if predictor_params is not None:
        predictor_params_.update(predictor_params)

    predictor = BaselinePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip, infer_size =infer_size, **predictor_params_)

    return predictor



def main():
    args, cfg = parse_args()

    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    print('checkpoint list: ', checkpoints_list)
    logs_path.mkdir(parents=True, exist_ok=True)

    single_model_eval = len(checkpoints_list) == 1
    assert not args.iou_analysis if not single_model_eval else True, \
        "Can't perform IoU analysis for multiple checkpoints"
    print_header = single_model_eval
    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg, args)

        for checkpoint_path in checkpoints_list:
            model = utils.load_is_model(checkpoint_path, args.device)

            predictor_params, zoomin_params, infer_size = get_predictor_and_zoomin_params(args, dataset_name)
            predictor = get_predictor(model, args.mode, args.device,
                                      infer_size=infer_size,
                                      prob_thresh=args.thresh,
                                      predictor_params=predictor_params,
                                      focus_crop_r = args.focus_crop_r,
                                      zoom_in_params=zoomin_params)
            
            vis_callback = get_prediction_vis_callback(logs_path, dataset_name, args.thresh) if args.vis_preds else None
            dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
                                               max_iou_thr=args.target_iou,
                                               min_clicks=args.min_n_clicks,
                                               max_clicks=args.n_clicks,
                                               callback=vis_callback, args=args)

            row_name = args.mode if single_model_eval else checkpoint_path.stem
            if args.iou_analysis:
                save_iou_analysis_data(args, dataset_name, logs_path,
                                       logs_prefix, dataset_results,
                                       model_name=args.model_name)

            save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                         save_ious=single_model_eval and args.save_ious,
                         single_model_eval=single_model_eval,
                         print_header=print_header)
            print_header = False


def get_predictor_and_zoomin_params(args, dataset_name):
    predictor_params = {}

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    if args.eval_mode == 'cvpr':
        zoom_in_params = {
            # We force resize on first interaction since base model not able to handle high resolution TETRIS images
            'skip_clicks': -1 if dataset_name == 'TETRIS' else 1,
            'target_size': 600 if dataset_name == 'DAVIS' else 400,
            'expansion_ratio': args.target_crop_r
        }
    elif args.eval_mode.startswith('fixed'):
        crop_size = int(args.eval_mode[5:])
        zoom_in_params = {
            'skip_clicks': -1,
            'target_size': (crop_size, crop_size)
        }
    else:
        raise NotImplementedError

    if dataset_name == 'SBD':
        infer_size=256
    else:
        infer_size=384

    return predictor_params, zoom_in_params,infer_size


if __name__ == '__main__':
    main()
