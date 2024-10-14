import torch
from torch import nn
import torch.nn.functional as F
from . import paths
from mmseg.registry import MODELS as MODELS_SEG

# import sys
# prg_dir = '/home/jovyan/d.shepelev/clickability_benchmark'
# sys.path.append(prg_dir)
# from mmscan import get_mmscan
# from mmseg_utils import (
#         IsStemConv, 
#         ISCascadeEncoderDecoder, 
#         IsMSCAN
# )

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmengine.registry import MODELS
from mmseg.registry import MODELS as MODELS_SEG
from mmseg.models.backbones.mscan import MSCAN, StemConv
from mmseg.models.segmentors import BaseSegmentor, EncoderDecoder
from mmseg.models.segmentors import CascadeEncoderDecoder
from mmseg.models.decode_heads import DepthwiseSeparableASPPHead
from mmseg.models.utils import resize

from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.psp_head import PPM
from mmseg.models.decode_heads.uper_head import UPerHead
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .utils import padding, unpadding
from transalnet.TranSalNet_Res import TranSalNet


@MODELS.register_module(force=True)
class ISCascadeEncoderDecoder(CascadeEncoderDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        features = self.extract_feat(inputs)
        
        decode_head_out = self.decode_head[0].forward(features)
        for i in range(1, self.num_stages):
            # TODO support PointRend tensor mode
            decode_head_out = self.decode_head[i].forward(features, decode_head_out)
        
        aux_out = []
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_out.append(aux_head.forward(features))
            else:
                aux_out = [self.auxiliary_head.forward(features)]
        
        return {'output': decode_head_out, 'aux_output': aux_out, 'features': features}


@MODELS.register_module(force=True)
class IsStemConv(StemConv):
    def forward(self, x, additional_features=None):
        """Forward function."""

        x = self.proj(x)
        if additional_features is not None:
            #print('additional features here!')
            x = x + additional_features
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W



@MODELS.register_module(force=True)
class IsMSCAN(MSCAN):
    def __init__(self, *args, 
                 embed_dims=[64, 128, 256, 512], norm_cfg=dict(type='SyncBN', requires_grad=True), 
                 **kwargs):
        super().__init__(*args, embed_dims=embed_dims, norm_cfg=norm_cfg, **kwargs)
        setattr(self, f'patch_embed1', IsStemConv(3, embed_dims[0], norm_cfg=norm_cfg))

    def forward(self, inputs):
        """Forward function."""
        x, additional_features = inputs
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x, additional_features) if i == 0 else patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

@MODELS.register_module(force=True)
class IsDepthwiseSeparableASPPHead(DepthwiseSeparableASPPHead):
    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output_enc = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output_dec = resize(
                input=output_enc,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output_dec = torch.cat([output_dec, c1_output], dim=1)
            output = self.sep_bottleneck(output_dec)
        else:
            output = self.sep_bottleneck(output_enc)
        lr_mask = self.cls_seg(output)
        return lr_mask, output, output_enc

# mmpretrain's convnext requires `data_format` argument in 
# self.norm call (mmpretrain/models/backbones/convnext.py, lines 100, 108)
@MODELS.register_module(force=True)
class MMSEG_BN2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, data_format='channel_first', **kwargs):
        assert data_format == 'channel_first'
        return super().forward(*args, **kwargs)

@MODELS.register_module(force=True)
class MMSEG_SYNCBN(torch.nn.SyncBatchNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, data_format='channel_first', **kwargs):
        assert data_format == 'channel_first'
        return super().forward(*args, **kwargs)

class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
        # avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        # x = avg(x)
        x = F.avg_pool2d(x, kernel_size.tolist(), stride=stride_size.tolist())
        return x



class PPMCustom(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    # use AdaptiveAvgPool2dCustom instead of nn.AdaptiveAvgPool2d
                    # https://github.com/pytorch/pytorch/issues/42653#issuecomment-1168816422
                    AdaptiveAvgPool2dCustom(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        **kwargs)))
            
    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

@MODELS.register_module(force=True)    
class UPerHeadCustom(UPerHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(pool_scales, **kwargs)
        self.psp_modules = PPMCustom(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

def get_mmscan(arch_type='t'):
    assert arch_type in ['t', 's', 'b', 'l']
    checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth' 
    ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
    model= dict(
        type='ISCascadeEncoderDecoder',
        num_stages=1,
        # data_preprocessor=data_preprocessor,
        pretrained=None,
        backbone=dict(
            type='IsMSCAN',
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
            embed_dims=[32, 64, 160, 256],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0,
            drop_path_rate=0.1,
            depths=[3, 3, 5, 2],
            attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
            attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='BN', requires_grad=True)),
        decode_head=[dict(
            type='LightHamHead',
            # in_channels=[64, 160, 256],
            in_channels=[32, 64, 160, 256],
            # in_index=[1, 2, 3],
            in_index=[0, 1, 2, 3],
            channels=256,
            ham_channels=256,
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=ham_norm_cfg,
            align_corners=False,
            ham_kwargs=dict(
                MD_S=1,
                MD_R=16,
                train_steps=6,
                eval_steps=7,
                inv_t=100,
                rand_init=False))],
        )
    if arch_type == 's':
        checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth'
        ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        
        model['backbone']['embed_dims']=[64, 128, 320, 512]
        model['backbone']['depths']=[2, 2, 4, 2]
        model['backbone']['init_cfg']=dict(type='Pretrained', checkpoint=checkpoint_file)
        model['backbone']['norm_cfg']=dict(type='BN', requires_grad=True)
        # model['decode_head'][0]['in_channels']=[128, 320, 512]
        model['decode_head'][0]['in_channels']=[64, 128, 320, 512]
        model['decode_head'][0]['channels']=256
        model['decode_head'][0]['ham_channels']=256
        model['decode_head'][0]['ham_kwargs']=dict(MD_S=1,
                MD_R=16,
                train_steps=6,
                eval_steps=7,
                inv_t=100,
                rand_init=False)
        model['decode_head'][0]['dropout_ratio']=0.1
        model['decode_head'][0]['norm_cfg']=ham_norm_cfg
        model['decode_head'][0]['align_corners']=False
    elif arch_type == 'b':
        checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_b_20230227-3ab7d230.pth'
        model['backbone']['embed_dims'] = [64, 128, 320, 512]
        model['backbone']['depths'] = [3, 3, 12, 3]
        model['backbone']['init_cfg']= dict(type='Pretrained', checkpoint=checkpoint_file)
        model['backbone']['norm_cfg']= dict(type='BN', requires_grad=True)
        model['decode_head'][0]['in_channels'] = [64, 128, 320, 512]
        model['decode_head'][0]['channels'] = 512
        model['decode_head'][0]['ham_channels'] = 512
        model['decode_head'][0]['ham_kwargs']=dict(MD_S=1,
                MD_R=16,
                train_steps=6,
                eval_steps=7,
                inv_t=100,
                rand_init=False)
        model['decode_head'][0]['dropout_ratio'] = 0.1
        model['decode_head'][0]['norm_cfg'] = ham_norm_cfg
        model['decode_head'][0]['align_corners'] = False


    return model

class BatchImageNormalize:
    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()
        tensor.sub_(self.mean.to(tensor.device)).div_(self.std.to(tensor.device))
        return tensor

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale

class LRMult(object):
    def __init__(self, lr_mult=1.):
        self.lr_mult = lr_mult

    def __call__(self, m):
        if getattr(m, 'weight', None) is not None:
            m.weight.lr_mult = self.lr_mult
        if getattr(m, 'bias', None) is not None:
            m.bias.lr_mult = self.lr_mult

def batch_normalize_map(tensor, eps=1e-9):
    shapes = tensor.shape
    tensor = tensor.view(shapes[0], -1)
    tensor = (tensor - tensor.min(dim=1, keepdim=True).values) / (tensor.max(dim=1, keepdim=True).values - tensor.min(dim=1, keepdim=True).values + eps)
    tensor = tensor.view(shapes)
    return tensor

class SegNeXtSaliency(nn.Module):
    def __init__(self,
                 arch_type='b',
                 backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, 
                 use_leaky_relu=False,
                 coord_feature_ch=2,
                 norm_mean_std=([.485, .456, .406], [.229, .224, .225]),
                 with_prev_mask=False,
                 **kwargs):
        super(SegNeXtSaliency, self).__init__()
        self.norm_layer = norm_layer
        self.use_leaky_relu = use_leaky_relu
        self.with_prev_mask = with_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])
        encoder_decoder_config = get_mmscan(arch_type=arch_type)
        self.encoder_decoder = MODELS_SEG.build(encoder_decoder_config)
        self.encoder_decoder.backbone.apply(LRMult(backbone_lr_mult))
        self.coord_feature_ch = coord_feature_ch
        mt_layers = [
            nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=encoder_decoder_config['backbone']['embed_dims'][0], 
                      kernel_size=3, stride=2, padding=1),
            ScaleLayer(init_value=0.05, lr_mult=1)
        ]
        self.maps_transform = nn.Sequential(*mt_layers)
    
    def forward(self, image, coord_features):
#       image = self.prepare_input(image)
#       coord_features = self.get_coord_features(prev_mask, contour_mask)
        coord_features = self.maps_transform(coord_features)
#       print(coord_features.shape)
        outputs = self.encoder_decoder((image, coord_features))
        
        outputs['output'] = F.interpolate(outputs['output'], size=image.shape[2:], mode='bilinear', align_corners=True)
        
        return {'instances': batch_normalize_map(outputs['output'])}# F.log_softmax(outputs['output'].view(-1, 1, image.shape[2] * image.shape[3]), dim=2).view(-1, 1, image.shape[2], image.shape[3])}# torch.sigmoid(outputs['output'])}


class SegNeXtSaliencyApply(SegNeXtSaliency):
    def __init__(self,
                 path,
                 arch_type='b',
                 backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, 
                 use_leaky_relu=False,
                 coord_feature_ch=2,
                 norm_mean_std=([.485, .456, .406], [.229, .224, .225]),
                 with_prev_mask=False,
                 device='cuda',
                 **kwargs):
        super(SegNeXtSaliencyApply, self).__init__(
            arch_type=arch_type, backbone_lr_mult=backbone_lr_mult,
            norm_layer=norm_layer, use_leaky_relu=use_leaky_relu,
            coord_feature_ch=coord_feature_ch, norm_mean_std=norm_mean_std, with_prev_mask=with_prev_mask)
        self.path = path
        self.normalization = A.Compose([A.Normalize(mean=norm_mean_std[0], std=norm_mean_std[1], max_pixel_value=1.0)])
        self.forward(torch.randn(1, 3, 100, 100), torch.randn(1, 2, 100, 100))
        self.load_state_dict(torch.load(self.path))
        self.device = device
        for param in self.parameters():
            param.requires_grad = False
        self.to(self.device)

    
    def apply(self, image, mask, prev_mistake):
        
        image_old_shape = image.shape        
        
        image = image / 255.
        
        image = torch.Tensor(padding(self.normalization(image=image)['image']))[None].permute(0, 3, 1, 2)
                    
        mask = torch.from_numpy(padding(mask.astype(np.float32)))[None, None]
        prev_mistake = torch.from_numpy(padding(prev_mistake.astype(np.float32)))[None, None]

        coord_features = torch.cat((mask, prev_mistake), 1)
        coord_features = self.maps_transform(coord_features.to(self.device))
        
        outputs = self.encoder_decoder((image.to(self.device), coord_features))
        outputs['output'] = F.interpolate(outputs['output'], size=image.shape[2:], 
                                          mode='bilinear', align_corners=True)
        
        outputs['output'] = batch_normalize_map(outputs['output'], eps=1e-8)

        outputs['output'] = unpadding(outputs['output'], (image_old_shape[0], image_old_shape[1]))
        
        return outputs['output']

def load_segnext(
        fname,
        device='cuda'):
    model = SegNeXtSaliency().to(device)
    model(torch.randn(1, 3, 100, 100).to(device), torch.randn(1, 2, 100, 100).to(device))
    model.load_state_dict(torch.load(paths.MODELS_DIR / fname))
    model.eval();
    return model

def load_transalnet_model(device='cuda'):
    sal_model = TranSalNet().to(device)
    pth_path =  paths.BENCHMARK_DIR / 'transalnet/transalnet/pretrained_models/TranSalNet_Res.pth'
    sal_model.load_state_dict(torch.load(pth_path))
    sal_model.eval();
    return sal_model
