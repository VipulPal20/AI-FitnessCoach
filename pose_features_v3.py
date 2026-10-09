"""1.9-second, 20-sample normalized pose features for v3 movement inference.

Input timestamps are seconds and must cover the entire trailing window. Missing
coverage or a raw observation gap longer than 0.35 seconds is never filled in.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from pose_features_v2 import FEATURE_SIZE, normalize_landmarks, temporal_summary

WINDOW_SECONDS = 1.9
SAMPLING_FPS = 10
SAMPLES = 20
MAX_GAP_SECONDS = 0.35
TIME_EPSILON = 1e-5  # float32 timestamps from NPZ may round 1.9 by ~2e-8
LEFT_SIDE = (11, 13, 15, 23, 25, 27)
RIGHT_SIDE = (12, 14, 16, 24, 26, 28)


def landmarks_to_xyz(
    landmarks: Sequence[Any] | None, visibility_threshold: float = 0.35
) -> np.ndarray | None:
    """Gate one MediaPipe frame; return (33, 3) XYZ or None on bad visibility.

    Visibility zero is an actual failed measurement, never a missing-value
    shorthand. All coordinates must be present and finite, but only one full
    arm/leg side must pass the visibility threshold.
    """

    if not math.isfinite(visibility_threshold) or not 0 <= visibility_threshold <= 1:
        raise ValueError("visibility_threshold must be finite and within [0, 1]")
    if landmarks is None or len(landmarks) != 33:
        return None
    xyz = np.empty((33, 3), dtype=np.float64)
    visible = np.zeros(33, dtype=bool)
    for index, landmark in enumerate(landmarks):
        try:
            coordinates = (float(landmark.x), float(landmark.y), float(landmark.z))
            visibility = float(landmark.visibility)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in (*coordinates, visibility)):
            return None
        xyz[index] = coordinates
        visible[index] = visibility >= visibility_threshold
    if not (visible[list(LEFT_SIDE)].all() or visible[list(RIGHT_SIDE)].all()):
        return None
    return xyz


def _validated_window(xyz: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(xyz, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (33, 3):
        raise ValueError("xyz must have shape (frames, 33, 3)")
    if times.ndim != 1 or len(times) != len(poses) or len(times) < 2:
        raise ValueError("timestamps must be a 1D array matching at least two frames")
    if not np.isfinite(poses).all() or not np.isfinite(times).all():
        raise ValueError("xyz and timestamps must be finite")
    gaps = np.diff(times)
    if np.any(gaps <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return poses, times


def extract_window_features(xyz: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Interpolate normalized XYZ onto 20 points over the trailing 1.9 seconds.

    Normalize *before* interpolation, using the same v2 hip/torso method.
    Ignore old observations preceding the trailing window except for the one
    bracketing its start; reject any long gap intersecting the chosen window.
    """

    poses, times = _validated_window(xyz, timestamps)
    start = times[-1] - WINDOW_SECONDS
    if times[0] > start + TIME_EPSILON:
        raise ValueError("Input must cover the full 1.9-second trailing window")
    left = max(0, int(np.searchsorted(times, start, side="right")) - 1)
    if np.any(np.diff(times[left:]) > MAX_GAP_SECONDS + 1e-9):
        raise ValueError("A pose observation gap exceeds 0.35 seconds")
    normalized = normalize_landmarks(poses[left:])
    sample_times = start + np.arange(SAMPLES, dtype=np.float64) / SAMPLING_FPS
    sample_times[-1] = times[-1]  # avoid floating-point drift at the endpoint
    sampled = np.empty((SAMPLES, 33, 3), dtype=np.float64)
    flattened = normalized.reshape(len(normalized), -1)
    for coordinate in range(flattened.shape[1]):
        sampled[:, coordinate // 3, coordinate % 3] = np.interp(
            sample_times, times[left:], flattened[:, coordinate]
        )
    result = temporal_summary(sampled)
    assert result.shape == (FEATURE_SIZE,)
    return result
