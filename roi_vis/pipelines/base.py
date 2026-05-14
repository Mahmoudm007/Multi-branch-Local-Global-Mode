"""Shared pipeline result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class PipelineArtifact:
    """A single saveable output."""

    key: str
    data: np.ndarray


@dataclass
class PipelineResult:
    """Outputs and metrics from a pipeline."""

    name: str
    artifacts: List[PipelineArtifact] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def add_artifact(self, key: str, data: np.ndarray) -> None:
        """Append an output artifact."""

        self.artifacts.append(PipelineArtifact(key=key, data=data))
