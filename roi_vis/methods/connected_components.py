"""Connected-component helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils.mask_utils import to_mask_uint8


@dataclass
class ComponentRecord:
    """Simple component statistics."""

    label: int
    area: int
    x: int
    y: int
    width: int
    height: int



def extract_component_records(mask: np.ndarray) -> list[ComponentRecord]:
    """Return connected-component records for a binary mask."""

    src = to_mask_uint8(mask)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(src, 8)
    records: list[ComponentRecord] = []
    for label in range(1, num_labels):
        records.append(
            ComponentRecord(
                label=label,
                area=int(stats[label, cv2.CC_STAT_AREA]),
                x=int(stats[label, cv2.CC_STAT_LEFT]),
                y=int(stats[label, cv2.CC_STAT_TOP]),
                width=int(stats[label, cv2.CC_STAT_WIDTH]),
                height=int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
        )
    return records



def max_component_area(mask: np.ndarray) -> int:
    """Return the largest component area."""

    records = extract_component_records(mask)
    if not records:
        return 0
    return max(record.area for record in records)
