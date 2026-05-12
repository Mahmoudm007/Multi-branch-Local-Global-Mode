from __future__ import annotations

import torch
from torch import nn

from .experiment_registry import ORIGINAL


class MultiBranchFusion(nn.Module):
    def __init__(
        self,
        visual_branches: tuple[str, ...],
        hidden_dim: int,
        mode: str = "gated",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if ORIGINAL not in visual_branches:
            raise ValueError("The original branch must be active for fusion")
        if mode not in {"concat", "gated", "film"}:
            raise ValueError(f"Unsupported fusion mode: {mode}")
        self.visual_branches = tuple(visual_branches)
        self.supplemental = tuple(branch for branch in visual_branches if branch != ORIGINAL)
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.gates = nn.ModuleDict(
            {
                branch: nn.Sequential(
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid(),
                )
                for branch in self.supplemental
            }
        )
        self.film = nn.ModuleDict(
            {
                branch: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                )
                for branch in self.supplemental
            }
        )
        fused_dim = hidden_dim * len(self.visual_branches)
        self.context = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, branch_embeddings: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        anchor = branch_embeddings[ORIGINAL]
        fused_parts = [anchor]
        gates: dict[str, torch.Tensor] = {}
        for branch in self.supplemental:
            current = branch_embeddings[branch]
            if self.mode == "concat":
                fused_parts.append(current)
            elif self.mode == "gated":
                gate = self.gates[branch](torch.cat([anchor, current], dim=1))
                gates[branch] = gate
                fused_parts.append(current * gate)
            else:
                gamma, beta = torch.chunk(self.film[branch](current), 2, dim=1)
                gamma = 0.1 * torch.tanh(gamma)
                beta = 0.1 * torch.tanh(beta)
                gates[branch] = gamma
                fused_parts.append(anchor * (1.0 + gamma) + beta)
        fused_raw = torch.cat(fused_parts, dim=1)
        return self.context(fused_raw), gates

