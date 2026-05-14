"""Technique-specific output folder and filename prefix definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactSpec:
    """Where and how to save an artifact."""

    key: str
    folder: str
    prefix: str
    file_type: str
    default_ext: str


ARTIFACT_SPECS = {
    'sky_mask': ArtifactSpec('sky_mask', 'SKY_MASK', 'SKYMASK', 'mask', '.png'),
    'sky_removed': ArtifactSpec('sky_removed', 'SKY_REMOVED', 'SKYREM', 'image', '.jpg'),
    'horizon_overlay': ArtifactSpec('horizon_overlay', 'HORIZON_OVERLAY', 'HORIZON', 'image', '.jpg'),
    'road_roi_mask': ArtifactSpec('road_roi_mask', 'ROAD_ROI_MASK', 'ROADMASK', 'mask', '.png'),
    'road_only': ArtifactSpec('road_only', 'ROAD_ONLY', 'ROADROI', 'image', '.jpg'),
    'road_overlay': ArtifactSpec('road_overlay', 'DEBUG_OVERLAYS', 'ROADOVL', 'image', '.jpg'),
    'lane_mask': ArtifactSpec('lane_mask', 'LANE_MASK', 'LANEMASK', 'mask', '.png'),
    'lane_overlay': ArtifactSpec('lane_overlay', 'LANE_OVERLAY', 'LANEOVL', 'image', '.jpg'),
    'shadow_mask': ArtifactSpec('shadow_mask', 'SHADOW_MASK', 'SHADOWMASK', 'mask', '.png'),
    'shadow_suppressed': ArtifactSpec('shadow_suppressed', 'SHADOW_SUPPRESSED', 'SHADOWSUP', 'image', '.jpg'),
    'shadow_overlay': ArtifactSpec('shadow_overlay', 'DEBUG_OVERLAYS', 'SHADOWOVL', 'image', '.jpg'),
    'glare_mask': ArtifactSpec('glare_mask', 'GLARE_MASK', 'GLAREMASK', 'mask', '.png'),
    'glare_suppressed': ArtifactSpec('glare_suppressed', 'GLARE_SUPPRESSED', 'GLARESUP', 'image', '.jpg'),
    'glare_overlay': ArtifactSpec('glare_overlay', 'DEBUG_OVERLAYS', 'GLAREOVL', 'image', '.jpg'),
    'texture_map': ArtifactSpec('texture_map', 'TEXTURE_MAP', 'TEXTURE', 'image', '.png'),
    'entropy_map': ArtifactSpec('entropy_map', 'ENTROPY_MAP', 'ENTROPY', 'image', '.png'),
    'gradient_map': ArtifactSpec('gradient_map', 'GRADIENT_MAP', 'GRADIENT', 'image', '.png'),
    'superpixels': ArtifactSpec('superpixels', 'SUPERPIXELS', 'SUPERPIX', 'image', '.jpg'),
    'region_mask': ArtifactSpec('region_mask', 'REGION_MASKS', 'REGIONMASK', 'mask', '.png'),
    'nonroad_suppressed': ArtifactSpec('nonroad_suppressed', 'NONROAD_SUPPRESSED', 'NONROAD', 'image', '.jpg'),
    'snow_candidate': ArtifactSpec('snow_candidate', 'SNOW_CANDIDATE', 'SNOWMASK', 'mask', '.png'),
    'snow_refined': ArtifactSpec('snow_refined', 'SNOW_REFINED', 'SNOWREF', 'mask', '.png'),
    'snow_overlay': ArtifactSpec('snow_overlay', 'DEBUG_OVERLAYS', 'SNOWOVL', 'image', '.jpg'),
    'best_combined': ArtifactSpec('best_combined', 'BEST_COMBINED', 'BEST', 'image', '.jpg'),
    'combined_overlay': ArtifactSpec('combined_overlay', 'DEBUG_OVERLAYS', 'OVERLAY', 'image', '.jpg'),
    'road_crop': ArtifactSpec('road_crop', 'ROAD_CROP', 'ROADCROP', 'image', '.jpg'),
}



def get_artifact_spec(key: str) -> ArtifactSpec:
    """Resolve an artifact spec by key."""

    if key not in ARTIFACT_SPECS:
        raise KeyError(f"Unknown artifact key '{key}'.")
    return ARTIFACT_SPECS[key]
