import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchBackbone(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.cnn(patch).flatten(1)


class RefinementNet(nn.Module):
    def __init__(
        self,
        patch_size: int = 32,
        context_dim: int = 0,
        use_adapter: bool = False,
        ablation: str = "none",
        drop_patch_embedding: bool = False,
        drop_ellipsoid_embedding: bool = False,
        drop_context_embedding: bool = False,
        disable_param_gate: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.context_dim = context_dim
        self.use_adapter = use_adapter
        self.ablation = ablation
        self.use_patch_embedding = ablation != "A1" and not drop_patch_embedding
        self.use_ellipsoid_embedding = ablation != "A2" and not drop_ellipsoid_embedding
        self.use_context_embedding = ablation != "A5" and not drop_context_embedding
        self.use_param_gate = ablation != "A3" and not disable_param_gate
        self.patch_dim = 96 * 4 * 4
        self.param_dim = 64
        self.context_feat_dim = 64
        self.backbone = PatchBackbone(in_channels=2)
        self.param_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        if use_adapter:
            self.context_mlp = nn.Sequential(
                nn.Linear(context_dim, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
            )
            fusion_in = 0
            if self.use_patch_embedding:
                fusion_in += self.patch_dim
            if self.use_ellipsoid_embedding:
                fusion_in += self.param_dim
            if self.use_context_embedding:
                fusion_in += self.context_feat_dim
            self.shared_fusion = nn.Sequential(
                nn.Linear(fusion_in, 256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.GELU(),
            )
            self.adapter = nn.Sequential(
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, self.patch_dim),
                nn.Tanh(),
            )
            if self.use_param_gate:
                self.gate_head = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Linear(128, 6),
                )
            else:
                self.gate_head = None
            head_in = 256
            if self.use_patch_embedding:
                head_in += self.patch_dim
            if self.use_ellipsoid_embedding:
                head_in += self.param_dim
            if self.use_context_embedding:
                head_in += self.context_feat_dim
        else:
            self.context_mlp = None
            self.shared_fusion = None
            self.adapter = None
            self.gate_head = None
            head_in = 0
            if self.use_patch_embedding:
                head_in += self.patch_dim
            if self.use_ellipsoid_embedding:
                head_in += self.param_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        self.max_delta_mu_t = 0.08
        self.max_delta_sigma_t = 0.20
        self.max_delta_mu_f = 0.04
        self.max_delta_sigma_f = 0.08
        self.max_delta_theta = math.pi / 6
        self.max_delta_amp = 0.25

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, patch: torch.Tensor, init_params: torch.Tensor, context: torch.Tensor | None = None):
        feat = self.backbone(patch)
        param_feat = self.param_mlp(init_params)
        if not self.use_patch_embedding:
            feat = torch.zeros_like(feat)
        if not self.use_ellipsoid_embedding:
            param_feat = torch.zeros_like(param_feat)
        if self.use_adapter:
            if context is None:
                raise ValueError("context is required when use_adapter=True")
            context_feat = self.context_mlp(context)
            if not self.use_context_embedding:
                context_feat = torch.zeros_like(context_feat)
            fusion_inputs = []
            if self.use_patch_embedding:
                fusion_inputs.append(feat)
            if self.use_ellipsoid_embedding:
                fusion_inputs.append(param_feat)
            if self.use_context_embedding:
                fusion_inputs.append(context_feat)
            fused = self.shared_fusion(torch.cat(fusion_inputs, dim=1))
            if self.use_patch_embedding:
                feature_gate = 1.0 + 0.15 * self.adapter(fused)
                feat = feat * feature_gate
            if self.use_param_gate:
                param_gate = torch.sigmoid(self.gate_head(fused))
            else:
                param_gate = torch.ones_like(init_params)
            head_inputs = []
            if self.use_patch_embedding:
                head_inputs.append(feat)
            if self.use_ellipsoid_embedding:
                head_inputs.append(param_feat)
            if self.use_context_embedding:
                head_inputs.append(context_feat)
            head_inputs.append(fused)
            head_input = torch.cat(head_inputs, dim=1)
        else:
            head_inputs = []
            if self.use_patch_embedding:
                head_inputs.append(feat)
            if self.use_ellipsoid_embedding:
                head_inputs.append(param_feat)
            head_input = torch.cat(head_inputs, dim=1)
            param_gate = torch.ones_like(init_params)
        delta = self.head(head_input)

        delta_mu_t = torch.tanh(delta[:, 0]) * self.max_delta_mu_t * param_gate[:, 0]
        delta_mu_f = torch.tanh(delta[:, 1]) * self.max_delta_mu_f * param_gate[:, 1]
        delta_log_sigma_t = torch.tanh(delta[:, 2]) * self.max_delta_sigma_t * param_gate[:, 2]
        delta_log_sigma_f = torch.tanh(delta[:, 3]) * self.max_delta_sigma_f * param_gate[:, 3]
        delta_theta = torch.tanh(delta[:, 4]) * self.max_delta_theta * param_gate[:, 4]
        delta_amp = torch.tanh(delta[:, 5]) * self.max_delta_amp * param_gate[:, 5]

        refined = torch.stack(
            [
                (init_params[:, 0] + delta_mu_t).clamp(0.0, 1.0),
                (init_params[:, 1] + delta_mu_f).clamp(0.0, 1.0),
                (init_params[:, 2] * torch.exp(delta_log_sigma_t)).clamp(0.01, 0.5),
                (init_params[:, 3] * torch.exp(delta_log_sigma_f)).clamp(0.01, 0.5),
                (init_params[:, 4] + delta_theta).clamp(-math.pi / 2, math.pi / 2),
                (init_params[:, 5] + delta_amp).clamp(0.05, 1.0),
            ],
            dim=1,
        )
        extra = {"raw_delta": delta, "param_gate": param_gate}
        return refined, extra


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
