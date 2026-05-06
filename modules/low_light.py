"""Adaptive low-light frame enhancement using CLAHE on the LAB L-channel.

Only activates when mean frame luminance falls below a configurable threshold,
so performance cost is zero in well-lit environments.
"""

from __future__ import annotations

import cv2
import numpy as np

import modules.globals

# LAB L-channel range is 0-255 for uint8.
# Luminance below this value → enhancement activates.
_BRIGHTNESS_THRESHOLD = 85
# Pre-build CLAHE objects at three aggression levels to avoid per-frame allocation.
_CLAHE_MILD   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_CLAHE_MEDIUM = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
_CLAHE_STRONG = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))


def _pick_clahe(mean_l: float) -> cv2.CLAHE:
    """Return the CLAHE instance appropriate for the measured luminance."""
    ratio = mean_l / _BRIGHTNESS_THRESHOLD  # 1.0 = threshold, 0.0 = black
    if ratio > 0.70:
        return _CLAHE_MILD
    if ratio > 0.40:
        return _CLAHE_MEDIUM
    return _CLAHE_STRONG


def apply_low_light_enhancement(frame: np.ndarray) -> np.ndarray:
    """Apply adaptive CLAHE when the frame is darker than the brightness threshold.

    Enhancement is skipped entirely on bright frames, so there is no overhead in
    well-lit conditions.

    Args:
        frame: BGR uint8 image.

    Returns:
        Enhanced BGR uint8 image (may be the same object if skipped).
    """
    if not getattr(modules.globals, "low_light_mode", False):
        return frame

    if frame is None or frame.size == 0:
        return frame

    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        mean_l = float(np.mean(l_channel))

        if mean_l >= _BRIGHTNESS_THRESHOLD:
            return frame  # Bright enough — skip processing.

        clahe = _pick_clahe(mean_l)
        lab[:, :, 0] = clahe.apply(l_channel)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    except cv2.error:
        return frame  # Never crash the pipeline on an enhancement failure.
