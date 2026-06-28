from dataclasses import dataclass, field
import numpy as np


@dataclass
class AlignedVolume:
    files: list
    homographies: np.ndarray
    bounding_boxes: np.ndarray
    panorama_size: np.ndarray
    global_offset: np.ndarray
    w: int
    h: int
    path: np.ndarray
    warnings: list = field(default_factory=list)
