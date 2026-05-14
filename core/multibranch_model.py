from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .experiment_registry import ORIGINAL, TIMM_MODEL_CANDIDATES, VISUAL_BRANCHES, ExperimentSpec
from .multibranch_fusion import MultiBranchFusion


@dataclass
class BackboneBuildResult:
    model: nn.Module | None
    visual_dim: int
    timm_model_name: str | None
    pretrained: bool
    status: str
    error: str = ""


def _timm_available() -> bool:
    return importlib.util.find_spec("timm") is not None


def list_available_timm_models() -> set[str]:
    if not _timm_available():
        return set()
    import timm

    return set(timm.list_models())


def resolve_timm_model_name(family: str, available: set[str] | None = None) -> str | None:
    available = available if available is not None else list_available_timm_models()
    for candidate in TIMM_MODEL_CANDIDATES.get(family, (family,)):
        if candidate in available:
            return candidate
    # Last-resort fuzzy match for environments with renamed variants.
    normalized = family.replace("_", "").lower()
    for name in sorted(available):
        if normalized in name.replace("_", "").lower():
            return name
    return None


def build_timm_backbone(
    family: str,
    pretrained_requested: bool,
    allow_random_fallback: bool,
    allow_remote_download: bool = False,
) -> BackboneBuildResult:
    if not _timm_available():
        return BackboneBuildResult(None, 0, None, False, "unavailable", "timm is not installed")
    import timm

    available = set(timm.list_models())
    model_name = resolve_timm_model_name(family, available)
    if model_name is None:
        return BackboneBuildResult(None, 0, None, False, "unavailable", f"No local timm candidate for {family}")
    errors: list[str] = []
    if pretrained_requested and allow_remote_download:
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
            visual_dim = int(getattr(model, "num_features", 0) or getattr(model, "num_classes", 0) or 0)
            if visual_dim <= 0:
                visual_dim = _infer_visual_dim(model)
            return BackboneBuildResult(model, visual_dim, model_name, True, "available_pretrained")
        except Exception as exc:  # noqa: BLE001 - fallback is explicitly configured
            errors.append(f"pretrained failed: {exc}")
            if not allow_random_fallback:
                return BackboneBuildResult(None, 0, model_name, False, "unavailable", " | ".join(errors))
    elif pretrained_requested and not allow_remote_download:
        errors.append("pretrained requested but remote downloads are disabled; using random init fallback")
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="avg")
        visual_dim = int(getattr(model, "num_features", 0) or 0)
        if visual_dim <= 0:
            visual_dim = _infer_visual_dim(model)
        status = "available_random_init" if errors else "available"
        return BackboneBuildResult(model, visual_dim, model_name, False, status, " | ".join(errors))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"random init failed: {exc}")
        return BackboneBuildResult(None, 0, model_name, False, "unavailable", " | ".join(errors))


def _infer_visual_dim(model: nn.Module) -> int:
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        output = model(dummy)
        if isinstance(output, (tuple, list)):
            output = [item for item in output if isinstance(item, torch.Tensor)][-1]
        if output.ndim > 2:
            if output.ndim == 4:
                output = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
            else:
                output = output.flatten(1)
        return int(output.shape[1])


class FiveBranchExperimentModel(nn.Module):
    def __init__(
        self,
        backbone_family: str,
        experiment: ExperimentSpec,
        num_classes: int,
        hidden_dim: int = 512,
        fusion_mode: str = "gated",
        dropout: float = 0.2,
        aux_feature_dim: int = 0,
        aux_hidden_dim: int = 512,
        aux_align_dim: int = 256,
        separate_branch_backbones: bool = False,
        pretrained_requested: bool = True,
        allow_random_fallback: bool = True,
        allow_remote_download: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_family = backbone_family
        self.experiment = experiment
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.aux_feature_dim = aux_feature_dim
        self.aux_align_dim = aux_align_dim
        self.separate_branch_backbones = separate_branch_backbones

        build = build_timm_backbone(backbone_family, pretrained_requested, allow_random_fallback, allow_remote_download)
        if build.model is None:
            raise RuntimeError(build.error or f"Could not build backbone family {backbone_family}")
        self.visual_dim = build.visual_dim
        self.backbone_build = build
        visual_branches = experiment.visual_branches
        if separate_branch_backbones:
            self.branch_backbones = nn.ModuleDict({ORIGINAL: build.model})
            for branch in visual_branches:
                if branch == ORIGINAL:
                    continue
                branch_build = build_timm_backbone(backbone_family, pretrained_requested, allow_random_fallback, allow_remote_download)
                if branch_build.model is None:
                    raise RuntimeError(branch_build.error or f"Could not build branch backbone for {branch}")
                self.branch_backbones[branch] = branch_build.model
            self.shared_backbone = None
        else:
            self.shared_backbone = build.model
            self.branch_backbones = nn.ModuleDict()

        self.projections = nn.ModuleDict(
            {
                branch: nn.Sequential(
                    nn.LayerNorm(self.visual_dim),
                    nn.Linear(self.visual_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for branch in visual_branches
            }
        )
        self.fusion = MultiBranchFusion(visual_branches, hidden_dim, mode=fusion_mode, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.aux_encoder: nn.Module | None = None
        self.image_align_head: nn.Module | None = None
        self.aux_align_head: nn.Module | None = None
        if experiment.uses_aux_text and aux_feature_dim > 0:
            self.aux_encoder = nn.Sequential(
                nn.LayerNorm(aux_feature_dim),
                nn.Linear(aux_feature_dim, aux_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.image_align_head = nn.Sequential(
                nn.Linear(hidden_dim, aux_align_dim),
                nn.GELU(),
                nn.Linear(aux_align_dim, aux_align_dim),
            )
            self.aux_align_head = nn.Sequential(
                nn.Linear(hidden_dim, aux_align_dim),
                nn.GELU(),
                nn.Linear(aux_align_dim, aux_align_dim),
            )

    def _encode_raw(self, branch: str, images: torch.Tensor) -> torch.Tensor:
        backbone = self.branch_backbones[branch] if self.separate_branch_backbones else self.shared_backbone
        if backbone is None:
            raise RuntimeError("No visual backbone is configured")
        output = backbone(images)
        if isinstance(output, (list, tuple)):
            tensors = [item for item in output if isinstance(item, torch.Tensor)]
            output = tensors[-1]
        if output.ndim == 4:
            output = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
        elif output.ndim > 2:
            output = output.flatten(1)
        return output

    def extract_analysis_features(
        self,
        images: dict[str, torch.Tensor],
        aux_features: torch.Tensor | None = None,
        aux_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        raw: dict[str, torch.Tensor] = {}
        projected: dict[str, torch.Tensor] = {}
        for branch in self.experiment.visual_branches:
            raw[branch] = self._encode_raw(branch, images[branch])
            projected[branch] = self.projections[branch](raw[branch])
        fused_visual, branch_gates = self.fusion(projected)
        logits = self.classifier(fused_visual)
        features: dict[str, Any] = {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=1),
            "fused_visual_embedding": fused_visual,
            "branch_gates": branch_gates,
        }
        for branch, value in raw.items():
            features[f"{branch}_embedding"] = value
        for branch, value in projected.items():
            features[f"{branch}_projected_embedding"] = value
        if self.aux_encoder is not None and aux_features is not None and aux_features.numel() > 0:
            features["aux_embedding"] = self.aux_encoder(aux_features)
            features["aux_input"] = aux_features
            if aux_mask is None:
                aux_mask = torch.ones(aux_features.shape[0], device=aux_features.device, dtype=aux_features.dtype)
            features["aux_mask"] = aux_mask.to(device=aux_features.device, dtype=aux_features.dtype).view(-1)
        return features

    def forward(
        self,
        images: dict[str, torch.Tensor],
        aux_features: torch.Tensor | None = None,
        aux_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.extract_analysis_features(images, aux_features, aux_mask)["logits"]

    def auxiliary_alignment_loss(self, features: dict[str, Any]) -> torch.Tensor:
        visual = features.get("fused_visual_embedding")
        aux = features.get("aux_embedding")
        if not isinstance(visual, torch.Tensor) or not isinstance(aux, torch.Tensor):
            if isinstance(visual, torch.Tensor):
                return visual.new_tensor(0.0)
            return torch.tensor(0.0)
        aux_mask = features.get("aux_mask")
        if isinstance(aux_mask, torch.Tensor):
            valid = aux_mask.to(device=visual.device).view(-1) > 0.5
        else:
            valid = torch.ones(visual.shape[0], device=visual.device, dtype=torch.bool)
        if not bool(valid.any()):
            return visual.new_tensor(0.0)
        if self.image_align_head is not None and self.aux_align_head is not None:
            image_embedding = F.normalize(self.image_align_head(visual[valid]), dim=1)
            aux_embedding = F.normalize(self.aux_align_head(aux[valid]), dim=1)
        else:
            image_embedding = F.normalize(visual[valid], dim=1)
            aux_embedding = F.normalize(aux[valid], dim=1)
        return 1.0 - (image_embedding * aux_embedding).sum(dim=1).mean()

    def metadata(self) -> dict[str, Any]:
        return {
            "backbone_family": self.backbone_family,
            "timm_model_name": self.backbone_build.timm_model_name,
            "backbone_status": self.backbone_build.status,
            "pretrained": self.backbone_build.pretrained,
            "visual_dim": self.visual_dim,
            "hidden_dim": self.hidden_dim,
            "branches": list(self.experiment.branches),
            "visual_branches": list(self.experiment.visual_branches),
            "uses_aux_text": self.experiment.uses_aux_text,
            "aux_feature_dim": self.aux_feature_dim,
            "aux_align_dim": self.aux_align_dim,
            "separate_branch_backbones": self.separate_branch_backbones,
            "shared_backbone": not self.separate_branch_backbones,
            "logits_are_image_driven": True,
        }
