import torch
from torch import nn as nn
import numpy as np
import isegm.model.initializer as initializer


def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU
        elif activation.lower() == 'softplus':
            return nn.Softplus
        else:
            raise ValueError(f"Unknown activation type {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Unknown activation type {activation}")


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, scale, groups=1):
        kernel_size = 2 * scale - scale % 2
        self.scale = scale

        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=scale,
            padding=1,
            groups=groups,
            bias=False)

        self.apply(initializer.Bilinear(scale=scale, in_channels=in_channels, groups=groups))


class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            from isegm.utils.cython import get_dist_maps
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                  norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)

            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
            coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])



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


class BatchImageNormalize:
    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()

        tensor.sub_(self.mean.to(tensor.device)).div_(self.std.to(tensor.device))
        return tensor


def DetrministicBilinear2D(input, size, align_corners=True):

    n_batch, n_channels, in_h, in_w = input.shape

    if align_corners:
        h_scale_factor = (in_h - 1) / (size[0] - 1)
        w_scale_factor = (in_w - 1) / (size[1] - 1)
    else:
        h_scale_factor = in_h / size[0]
        w_scale_factor = in_w / size[1]


    i = torch.arange(size[0], dtype=input.dtype, device=input.device)
    j = torch.arange(size[1], dtype=input.dtype, device=input.device)

    if align_corners:
        x = h_scale_factor * i
        y = w_scale_factor * j
    else:
        x = (h_scale_factor * (i + 0.5) - 0.5).clamp(min=0.0)
        y = (w_scale_factor * (j + 0.5) - 0.5).clamp(min=0.0)

    x_floor = torch.floor(x).to(torch.int64)
    x_ceil = torch.ceil(x).clamp(max=in_h - 1).to(torch.int64)
    y_floor = torch.floor(y).to(torch.int64)
    y_ceil = torch.ceil(y).clamp(max=in_w - 1).to(torch.int64)

    x_view = x.unsqueeze(1)
    x_floor_view = x_floor.unsqueeze(1)

    xscale2 = x_view - x_floor_view
    xscale1 = 1.0 - xscale2

    yscale2 = y - y_floor
    yscale1 = 1.0 - yscale2

    x_ceil_view = x_ceil.unsqueeze(1)

    v1 = input[:, :, x_floor_view, y_floor]
    v2 = input[:, :, x_ceil_view, y_floor]
    v3 = input[:, :, x_floor_view, y_ceil]
    v4 = input[:, :, x_ceil_view, y_ceil]

    result = (v1 * xscale1 + v2 * xscale2) * yscale1 + (v3 * xscale1 + v4 * xscale2) * yscale2

    return result




class DistMapsForSampling(DistMaps):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super().__init__(norm_radius, spatial_scale, cpu_mode, use_disks)

    def get_coord_features_new(self, batchsize, rows, cols, last_click=None):
        
        if last_click is not None:
            x, y = last_click
        else:
            x, y = self.last_click

        points = np.array([[x, y, 0], [-1, -1, -1]])[None]
        img_rendered = self.get_coord_features(torch.from_numpy(points), 1, rows, cols)[0][0]
        
        img_opposite = torch.zeros_like(img_rendered)
        if self.last_click_idx >= self.points.shape[1] // 2:
            # Negative click
            coords_new = torch.stack([img_opposite, img_rendered])[None]
        else:
            # Positive click
            coords_new = torch.stack([img_rendered, img_opposite])[None]
        coords_output = torch.clamp(self.coords_output + coords_new, 0, 1)

        return coords_output, coords_new


    def forward(self):
        return self.get_coord_features_new(self.x.shape[0], self.x.shape[2], self.x.shape[3])


    def forward_with_last_click(self, last_click):
        return self.get_coord_features_new(self.x.shape[0], self.x.shape[2], self.x.shape[3], last_click)


    def register(self, x, coords_in):
        coords = coords_in.clone()[:1]
        self.x = x
        self.points = coords
        self.last_click_idx = coords[:, :, 2].argmax()
        last_xy = coords[:, self.last_click_idx][0][:2]
        self.last_click = torch.tensor([last_xy[1], last_xy[0]]).float()
        self.last_number = float(coords[:, self.last_click_idx][0][-1])
        coords[:, self.last_click_idx] = torch.tensor([[-1., -1., -1.]], device=self.last_click.device, dtype=torch.float64)
        self.coords_output = self.get_coord_features(coords, 1, x.shape[2], x.shape[3]).to(self.last_click.device)


