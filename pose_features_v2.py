"""NumPy-only temporal features for MediaPipe Pose's 33 landmarks.

The functions in this module deliberately do not know anything about exercise
labels or model training.  A clip is an array shaped ``(frames, 33, 3)``
containing MediaPipe ``x, y, z`` coordinates.  Coordinates are translated to
the midpoint of the left and right hip landmarks (23 and 24) and scaled by a
torso/shoulder measure, with a deterministic fallback for degenerate poses.

The temporal descriptor concatenates per-coordinate mean, standard deviation,
minimum, maximum, first frame, last frame, mean velocity, and velocity
standard deviation.  It is therefore a fixed 792-element NumPy vector for
every non-empty clip (8 summaries * 33 landmarks * 3 coordinates).

This module has no MediaPipe, scikit-learn, or label-generation dependency.
"""

from __future__ import annotations

import numpy as np


MEDIAPIPE_LANDMARK_COUNT = 33
XYZ_DIMENSIONS = 3
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
FEATURE_BLOCKS = (
    "mean",
    "std",
    "min",
    "max",
    "first",
    "last",
    "velocity_mean",
    "velocity_std",
)
FEATURE_SIZE = len(FEATURE_BLOCKS) * MEDIAPIPE_LANDMARK_COUNT * XYZ_DIMENSIONS


def _validate_xyz(landmarks: np.ndarray) -> np.ndarray:
    """Return a float64 clip after validating the MediaPipe XYZ shape."""

    values = np.asarray(landmarks, dtype=np.float64)
    if values.ndim == 2 and values.shape == (MEDIAPIPE_LANDMARK_COUNT, XYZ_DIMENSIONS):
        values = values[np.newaxis, ...]
    if values.ndim != 3 or values.shape[1:] != (
        MEDIAPIPE_LANDMARK_COUNT,
        XYZ_DIMENSIONS,
    ):
        raise ValueError(
            "Expected landmarks with shape (frames, 33, 3), "
            f"got {tuple(values.shape)}."
        )
    if values.shape[0] == 0:
        raise ValueError("A pose clip must contain at least one frame.")
    if not np.isfinite(values).all():
        raise ValueError("Pose coordinates must be finite; remove invalid frames first.")
    return values


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize a clip for translation and scale.

    The origin is the hip midpoint.  Scale is the distance between shoulder
    and hip midpoints; if that distance is zero, the largest landmark radius
    in the frame is used, and a final scale of one avoids division by zero.
    The returned array always has shape ``(frames, 33, 3)``.
    """

    values = _validate_xyz(landmarks)
    hip_midpoint = (values[:, LEFT_HIP_INDEX] + values[:, RIGHT_HIP_INDEX]) / 2.0
    shoulder_midpoint = (
        values[:, LEFT_SHOULDER_INDEX] + values[:, RIGHT_SHOULDER_INDEX]
    ) / 2.0
    scale = np.linalg.norm(shoulder_midpoint - hip_midpoint, axis=1)
    radii = np.linalg.norm(values - hip_midpoint[:, np.newaxis, :], axis=2)
    fallback = np.max(radii, axis=1)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, fallback)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    return (values - hip_midpoint[:, np.newaxis, :]) / scale[:, np.newaxis, np.newaxis]


def temporal_summary(normalized_clip: np.ndarray) -> np.ndarray:
    """Summarize a normalized clip into a deterministic one-dimensional vector."""

    values = _validate_xyz(normalized_clip)
    deltas = np.diff(values, axis=0)
    if deltas.shape[0] == 0:
        deltas = np.zeros((1,) + values.shape[1:], dtype=np.float64)
    blocks = (
        np.mean(values, axis=0),
        np.std(values, axis=0),
        np.min(values, axis=0),
        np.max(values, axis=0),
        values[0],
        values[-1],
        np.mean(deltas, axis=0),
        np.std(deltas, axis=0),
    )
    features = np.concatenate([block.reshape(-1) for block in blocks]).astype(
        np.float64, copy=False
    )
    if features.shape != (FEATURE_SIZE,) or not np.isfinite(features).all():
        raise ValueError("Temporal feature extraction produced an invalid descriptor.")
    return features


def extract_temporal_features(landmarks: np.ndarray) -> np.ndarray:
    """Normalize a MediaPipe XYZ clip and return its fixed-size temporal vector."""

    return temporal_summary(normalize_landmarks(landmarks))


# A concise alias for callers that prefer a pose-oriented name.
extract_pose_features = extract_temporal_features
