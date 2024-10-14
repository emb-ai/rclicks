from .base import BasePredictor
from isegm.inference.transforms import ZoomIn
from isegm.model.is_hrnet_model import HRNetModel


def get_predictor(net, brs_mode, device,
                  prob_thresh=0.49,
                  with_flip=True,
                  zoom_in_params=dict(),
                  predictor_params=None,
                  brs_opt_func_params=None,
                  lbfgs_params=None,
                  model_name='ritm'):
    lbfgs_params_ = {
        'm': 20,
        'factr': 0,
        'pgtol': 1e-8,
        'maxfun': 20,
    }

    predictor_params_ = {
        'optimize_after_n_clicks': 1
    }

    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    if lbfgs_params is not None:
        lbfgs_params_.update(lbfgs_params)
    lbfgs_params_['maxiter'] = 2 * lbfgs_params_['maxfun']

    if brs_opt_func_params is None:
        brs_opt_func_params = dict()

    if isinstance(net, (list, tuple)):
        assert brs_mode == 'NoBRS', "Multi-stage models support only NoBRS mode."

    if brs_mode == 'NoBRS':
        if predictor_params is not None:
            predictor_params_.update(predictor_params)
        predictor = BasePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip, model_name=model_name, **predictor_params_)
    else:
        raise NotImplementedError

    return predictor
