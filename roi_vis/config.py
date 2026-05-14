"""Central configuration for the classical preprocessing toolkit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ThresholdConfig:
    """Numeric thresholds used by the classical heuristics."""

    sky_top_ratio: float = 0.62
    sky_blue_margin: float = 8.0
    sky_low_texture_percentile: float = 55.0
    sky_low_chroma: float = 24.0
    sky_min_lightness: float = 135.0
    road_bottom_exclusion_ratio: float = 0.22
    road_seed_width_ratio: float = 0.12
    road_seed_y0_ratio: float = 0.60
    road_seed_y1_ratio: float = 0.80
    road_ab_distance: float = 24.0
    road_l_distance: float = 90.0
    lane_white_l: int = 168
    lane_white_s: int = 118
    lane_yellow_h_low: int = 12
    lane_yellow_h_high: int = 40
    glare_v_min: int = 220
    glare_s_max: int = 70
    glare_local_contrast: float = 12.0
    shadow_ratio: float = 0.72
    shadow_ab_distance: float = 24.0
    snow_l_min: int = 150
    snow_v_min: int = 170
    snow_chroma_max: float = 28.0
    snow_s_max: int = 86
    snow_texture_percentile: float = 68.0
    corridor_bottom_width_ratio: float = 0.92
    corridor_top_width_ratio: float = 0.22
    max_component_fraction: float = 0.35


@dataclass
class MorphologyConfig:
    """Morphological kernel sizes and component filtering."""

    open_size: int = 5
    close_size: int = 11
    dilate_size: int = 5
    erode_size: int = 3
    fill_size: int = 7
    min_component_area: int = 180
    min_lane_area: int = 60
    min_glare_area: int = 40
    min_shadow_area: int = 90
    min_snow_area: int = 120


@dataclass
class DatasetConfig:
    """Dataset structure configuration."""

    input_root: Path = Path('Dataset_Classes_v1')
    output_root: Path = Path('outputs')
    defined_folder_name: str = '1 Defined'
    ignore_folders: List[str] = field(default_factory=lambda: ['0 Undefined'])
    splits: List[str] = field(default_factory=lambda: ['train', 'val'])
    class_folder_names: List[str] = field(
        default_factory=lambda: [
            '0 Bare',
            '1 Centre_Partly',
            '4 Fully',
            '3 One_Track_Partly',
            '2 Two_Track_Partly',
        ]
    )
    class_aliases: Dict[str, List[str]] = field(
        default_factory=lambda: {
            '0 Bare': ['0 Bare'],
            '1 Centre_Partly': ['1 Centre_Partly', '1 Centre - Partly'],
            '2 Two_Track_Partly': ['2 Two_Track_Partly', '2 Two Track - Partly'],
            '3 One_Track_Partly': ['3 One_Track_Partly', '3 One Track - Partly'],
            '4 Fully': ['4 Fully'],
        }
    )
    image_extensions: Tuple[str, ...] = (
        '.jpg',
        '.jpeg',
        '.png',
        '.bmp',
        '.tif',
        '.tiff',
        '.webp',
    )


@dataclass
class ProjectConfig:
    """Top-level runtime configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    resize_width: int = 960
    resize_height: int = 720
    jpeg_quality: int = 95
    png_compression: int = 3
    debug: bool = False
    save_intermediate_outputs: bool = True
    max_images: int | None = None
    report_name: str = 'summary.csv'

    @property
    def resize_shape(self) -> Tuple[int, int]:
        return (self.resize_width, self.resize_height)



def create_default_config() -> ProjectConfig:
    """Create the default project configuration."""

    return ProjectConfig()



def create_config_from_args(args) -> ProjectConfig:
    """Create a runtime config from parsed CLI arguments."""

    cfg = create_default_config()
    cfg.dataset.input_root = Path(args.input)
    cfg.dataset.output_root = Path(args.output)
    cfg.dataset.defined_folder_name = args.defined_folder
    cfg.dataset.ignore_folders = list(args.ignore_folders)
    cfg.dataset.splits = list(args.splits)
    cfg.debug = bool(args.debug)
    cfg.save_intermediate_outputs = not bool(args.no_intermediate)
    cfg.max_images = args.max_images
    if args.resize_width:
        cfg.resize_width = int(args.resize_width)
    if args.resize_height:
        cfg.resize_height = int(args.resize_height)
    return cfg
