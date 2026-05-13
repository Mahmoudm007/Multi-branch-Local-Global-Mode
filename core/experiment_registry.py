from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from typing import Iterable

try:
    from class_descriptions import normalize_class_name
except Exception:  # pragma: no cover - import fallback for unusual launch paths
    def normalize_class_name(value: str) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"^\s*\d+\s*", "", text)
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()


ORIGINAL = "original"
CROPPED = "cropped"
THERMAL = "thermal"
SEGMENTED = "segmented"
AUX_TEXT = "auxiliary_text"

VISUAL_BRANCHES = (ORIGINAL, CROPPED, THERMAL, SEGMENTED)
OPTIONAL_BRANCH_ORDER = (THERMAL, SEGMENTED, CROPPED, AUX_TEXT)
REQUIRED_BRANCHES = (ORIGINAL,)

BRANCH_DIRS = {
    CROPPED: "cropped_local",
    THERMAL: "thermal_clahe_inferno",
    SEGMENTED: "segmented_best_combined",
}

BRANCH_NAME_TOKENS = {
    ORIGINAL: "original",
    CROPPED: "cropped",
    THERMAL: "thermal",
    SEGMENTED: "segmented",
    AUX_TEXT: "auxtext",
}

BRANCH_DISPLAY_NAMES = {
    ORIGINAL: "Original RGB",
    CROPPED: "Cropped local",
    THERMAL: "CLAHE Inferno thermal",
    SEGMENTED: "Best-combined segmented",
    AUX_TEXT: "Auxiliary class-description text",
}

MODEL_FAMILIES = (
    "convnext",
    "convnext_v2",
    "maxvit",
    "deit_base",
    "seresnext50",
    "inception_v3",
    "xception",
    "beit_base",
    "pvt_v2",
    "mambaout",
    "coatnet",
    "focalnet",
    "davit",
)

# Ordered to prefer base/large variants over tiny/small variants. The family name
# remains the benchmark identity; the chosen timm model is logged in the capability
# manifest.
TIMM_MODEL_CANDIDATES = {
    "convnext": ("convnext_base", "convnext_large", "convnext_large_mlp", "convnext_xlarge"),
    "convnext_v2": ("convnextv2_base", "convnextv2_large", "convnextv2_huge"),
    "maxvit": ("maxvit_base_tf_224", "maxvit_large_tf_224", "maxvit_rmlp_base_rw_224", "maxvit_xlarge_tf_224"),
    "deit_base": ("deit_base_patch16_224", "deit3_base_patch16_224", "deit3_large_patch16_224"),
    "seresnext50": ("seresnext50_32x4d", "legacy_seresnext50_32x4d"),
    "inception_v3": ("inception_v3", "tf_inception_v3"),
    "xception": ("xception65", "xception71", "xception41", "xception"),
    "beit_base": ("beit_base_patch16_224", "beitv2_base_patch16_224", "beit_large_patch16_224"),
    "pvt_v2": ("pvt_v2_b2", "pvt_v2_b3", "pvt_v2_b4", "pvt_v2_b5"),
    "mambaout": ("mambaout_base", "mambaout_large", "mambaout_kobe"),
    "coatnet": ("coatnet_2_rw_224", "coatnet_3_rw_224", "coatnet_4_224", "coatnet_5_224", "coatnet_1_rw_224"),
    "focalnet": ("focalnet_base_lrf", "focalnet_large_fl3", "focalnet_large_fl4", "focalnet_xlarge_fl3"),
    "davit": ("davit_base", "davit_large", "davit_base_fl", "davit_huge"),
}

CANONICAL_CLASS_ORDER = (
    "bare",
    "centre partly",
    "two track partly",
    "one track partly",
    "fully",
)

CANONICAL_CLASS_FOLDERS = (
    "0 Bare",
    "1 Centre - Partly",
    "2 Two Track - Partly",
    "3 One Track - Partly",
    "4 Fully",
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    branches: tuple[str, ...]

    @property
    def visual_branches(self) -> tuple[str, ...]:
        return tuple(branch for branch in self.branches if branch in VISUAL_BRANCHES)

    @property
    def uses_aux_text(self) -> bool:
        return AUX_TEXT in self.branches

    @property
    def branch_count(self) -> int:
        return len(self.branches)


def experiment_name(branches: Iterable[str]) -> str:
    return "exp_" + "_".join(BRANCH_NAME_TOKENS[branch] for branch in branches)


def build_experiments(include_original_only: bool = False) -> list[ExperimentSpec]:
    experiments: list[ExperimentSpec] = []
    priority_optionals = (
        (THERMAL, SEGMENTED, CROPPED, AUX_TEXT),
        (THERMAL, CROPPED, AUX_TEXT),
    )
    seen: set[tuple[str, ...]] = set()
    for optional in priority_optionals:
        branches = (ORIGINAL, *optional)
        experiments.append(ExperimentSpec(experiment_name(branches), branches))
        seen.add(optional)
    for optional_count in (1, 2, 3, 4):
        for optional in combinations(OPTIONAL_BRANCH_ORDER, optional_count):
            if optional in seen:
                continue
            branches = (ORIGINAL, *optional)
            experiments.append(ExperimentSpec(experiment_name(branches), branches))
    if include_original_only:
        experiments.append(ExperimentSpec("exp_original_only", (ORIGINAL,)))
    return experiments


def get_experiment(name: str, include_original_only: bool = True) -> ExperimentSpec:
    by_name = {spec.name: spec for spec in build_experiments(include_original_only)}
    if name not in by_name:
        valid = ", ".join(sorted(by_name))
        raise KeyError(f"Unknown experiment '{name}'. Valid experiments: {valid}")
    return by_name[name]


def normalize_dataset_class_name(folder_name: str) -> str:
    return normalize_class_name(folder_name)


def class_sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"^\s*(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name.lower()
    normalized = normalize_dataset_class_name(path.name)
    try:
        return CANONICAL_CLASS_ORDER.index(normalized), path.name.lower()
    except ValueError:
        return 999, path.name.lower()


def discover_class_folders(split_root: Path) -> list[Path]:
    folders = [p for p in split_root.iterdir() if p.is_dir()]
    return sorted(folders, key=class_sort_key)


def build_class_mapping(defined_root: Path) -> tuple[dict[str, int], dict[int, str], dict[str, str]]:
    train_root = defined_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Missing train split: {train_root}")
    class_folders = discover_class_folders(train_root)
    if not class_folders:
        raise FileNotFoundError(f"No class folders found under {train_root}")

    class_to_index: dict[str, int] = {}
    index_to_class: dict[int, str] = {}
    folder_to_class: dict[str, str] = {}
    for idx, folder in enumerate(class_folders):
        normalized = normalize_dataset_class_name(folder.name)
        class_to_index[normalized] = idx
        index_to_class[idx] = normalized
        folder_to_class[folder.name] = normalized
    return class_to_index, index_to_class, folder_to_class


def branch_asset_dir(asset_root: Path, branch: str, defined_folder: str = "1 Defined") -> Path:
    if branch not in BRANCH_DIRS:
        raise KeyError(f"Branch '{branch}' does not have generated image assets")
    return asset_root / BRANCH_DIRS[branch] / defined_folder


def ensure_original_in_experiment(spec: ExperimentSpec) -> None:
    if ORIGINAL not in spec.branches:
        raise ValueError(f"Experiment {spec.name} does not include the original branch")
