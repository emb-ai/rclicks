import sys
import torch
import numpy as np

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.inference.evaluation import evaluate_dataset
import os
import random
from isegm.inference.predictors.cdnet import DiffisionPredictor
from isegm.inference.transforms.zoom_in_gpcis import ZoomIn
from isegm.inference.utils import load_single_is_model
from evaluate_model_ritm import get_checkpoints_list_and_logs_path, save_results, save_iou_analysis_data, get_prediction_vis_callback
from evaluate_model_gpcis import parse_args


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

    if brs_opt_func_params is None:
        brs_opt_func_params = dict()

    if isinstance(net, (list, tuple)):
        assert brs_mode == 'NoBRS', "Multi-stage models support only NoBRS mode."

    
    if predictor_params is not None:
        predictor_params_.update(predictor_params)
    predictor = DiffisionPredictor(net, device, zoom_in=zoom_in, with_flip=with_flip, infer_size =infer_size, **predictor_params_)

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

            state_dict = torch.load(checkpoint_path, map_location='cpu')
            state_dict['config']['class'] = state_dict['config']['class'].replace('is_deeplab_model', 'is_cdnet_deeplab_model')
            model = load_single_is_model(state_dict, args.device)

            predictor_params, zoomin_params = get_predictor_and_zoomin_params(args, dataset_name)
            predictor = get_predictor(model, args.mode, args.device,
                                      infer_size=args.infer_size,
                                      prob_thresh=args.thresh,
                                      predictor_params=predictor_params,
                                      focus_crop_r = args.focus_crop_r,
                                      #zoom_in_params=None)
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

    return predictor_params, zoom_in_params


if __name__ == '__main__':
    main()
