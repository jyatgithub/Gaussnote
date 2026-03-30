import math

import torch
import torch.nn as nn


class GaussianEllipsoidRenderer(nn.Module):
    def __init__(self, patch_size: int = 32):
        super().__init__()
        axis = torch.linspace(0.0, 1.0, patch_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("t_grid", xx)
        self.register_buffer("f_grid", yy)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        mu_t = params[:, 0].view(-1, 1, 1)
        mu_f = params[:, 1].view(-1, 1, 1)
        sigma_t = params[:, 2].clamp_min(1e-3).view(-1, 1, 1)
        sigma_f = params[:, 3].clamp_min(1e-3).view(-1, 1, 1)
        theta = params[:, 4].clamp(-math.pi / 2, math.pi / 2).view(-1, 1, 1)
        amp = params[:, 5].clamp(0.0, 1.0).view(-1, 1, 1)

        dt = self.t_grid.unsqueeze(0) - mu_t
        df = self.f_grid.unsqueeze(0) - mu_f
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        u = cos_t * dt + sin_t * df
        v = -sin_t * dt + cos_t * df
        exponent = 0.5 * ((u ** 2) / (sigma_t ** 2) + (v ** 2) / (sigma_f ** 2))
        return amp * torch.exp(-exponent)
