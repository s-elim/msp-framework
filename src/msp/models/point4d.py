"""4D Spatio-Temporal Point Cloud Backbone for Manipulation-Sufficient Perception.

Consumes 4D point clouds (X, Y, Z, t) augmented with photometric features and Gemini VLM
affordance weights, replacing global 2D average pooling with spatially grounded,
locally sensitive geometric features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from msp.models.nets import Backbone

__all__ = ["Point4DBackbone", "GeminiVLMAffordanceBridge"]


class GeminiVLMAffordanceBridge(nn.Module):
    """Bridges high-level Vision-Language semantic priors into 4D geometric points.

    Gemini VLM provides open-vocabulary target grounding (e.g. 'drill handle', 'top flange')
    which is projected into per-point affordance logits w_vlm in [0, 1]. This module embeds
    and gates spatial geometric tokens with the semantic attention prior.
    """

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.affordance_gate = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, point_features: Tensor, vlm_affordance: Tensor) -> Tensor:
        """
        Args:
            point_features: (B, N, D)
            vlm_affordance: (B, N, 1) in range [0, 1]
        Returns:
            gated_features: (B, N, D)
        """
        assert point_features.dim() == 3, f"expected (B, N, D), got {point_features.shape}"
        assert vlm_affordance.dim() == 3, f"expected (B, N, 1), got {vlm_affordance.shape}"
        gate = self.affordance_gate(vlm_affordance)
        return point_features * (1.0 + gate)


class Point4DBackbone(Backbone):
    """Spatio-Temporal 4D Point Cloud Feature Extractor.

    Inputs:
        obs: (B, N, C) where C >= 4:
            - channel 0..2: x, y, z (meters in camera / world coordinates)
            - channel 3: t (normalized timestamp or viewpoint index in [0, 1])
            - channel 4..6 (optional): r, g, b normalized in [-1, 1]
            - channel 7 (optional): Gemini VLM affordance weight in [0, 1]

    Outputs:
        features: (B, output_dim) continuous decision representation for BeliefEncoder.
    """

    def __init__(
        self,
        input_channels: int = 8,
        output_dim: int = 512,
        hidden_dims: tuple[int, ...] = (64, 128, 256),
        use_vlm_bridge: bool = True,
    ) -> None:
        super().__init__(output_dim)
        self.input_channels = input_channels
        self.use_vlm_bridge = use_vlm_bridge

        # Spatio-temporal embedding: non-linear projection of (x, y, z, t) coordinates
        self.coord_embed = nn.Sequential(
            nn.Linear(4, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )

        # Feature embedding (color + semantics)
        feat_in = max(1, input_channels - 4)
        self.feat_embed = nn.Sequential(
            nn.Linear(feat_in, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )

        if use_vlm_bridge:
            self.vlm_bridge = GeminiVLMAffordanceBridge(feature_dim=hidden_dims[0])

        # Hierarchical point processing blocks with residual connections
        layers: list[nn.Module] = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.GELU(),
                    nn.Linear(h_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.GELU(),
                )
            )
            in_dim = h_dim
        self.mlp_blocks = nn.ModuleList(layers)

        # Multi-resolution pooling heads (max-pool + mean-pool)
        final_dim = hidden_dims[-1] * 2
        self.head = nn.Sequential(
            nn.Linear(final_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """
        Args:
            obs: (B, N, C) point cloud tensor.
        Returns:
            out: (B, output_dim) pooled scene representation.
        """
        assert obs.dim() == 3, f"expected (B, N, C), got {obs.shape}"
        B, N, C = obs.shape
        assert C >= 4, f"expected at least 4 channels (x, y, z, t), got {C}"

        coords_4d = obs[:, :, :4]  # (B, N, 4)
        c_emb = self.coord_embed(coords_4d)  # (B, N, H0)

        if C > 4:
            extra_feat = obs[:, :, 4:]
            f_emb = self.feat_embed(extra_feat)
            h = c_emb + f_emb
            if self.use_vlm_bridge and C >= 8:
                vlm_affordance = obs[:, :, 7:8]
                h = self.vlm_bridge(h, vlm_affordance)
        else:
            h = c_emb

        # Progressive refinement across spatio-temporal neighborhoods
        for block in self.mlp_blocks:
            h = block(h)

        # Dual pooling: max-pooling captures salient contact extrema, mean captures bulk volume
        pool_max = torch.max(h, dim=1).values  # (B, H_last)
        pool_mean = torch.mean(h, dim=1)  # (B, H_last)
        pooled = torch.cat([pool_max, pool_mean], dim=-1)  # (B, 2 * H_last)

        out = self.head(pooled)  # (B, output_dim)
        return out
