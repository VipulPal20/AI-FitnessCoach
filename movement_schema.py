"""Canonical movement labels and the v3 hierarchy shared by training and inference."""

from __future__ import annotations

ACTIONS = (
    "bench_press", "shoulder_press", "push_up", "squat", "deadlift",
    "bicep_curl", "pull_up", "row", "lunge", "jumping_jack", "unknown",
)

FAMILY_MAP = {
    "upper_push": ("bench_press", "shoulder_press", "push_up"),
    "upper_pull": ("pull_up", "row"),
    "elbow_flexion": ("bicep_curl",),
    "knee_dominant": ("squat", "lunge"),
    "hip_hinge": ("deadlift",),
    "whole_body": ("jumping_jack",),
    "unknown": ("unknown",),
}

ACTION_FAMILY = {action: family for family, actions in FAMILY_MAP.items() for action in actions}
assert set(ACTIONS) == set(ACTION_FAMILY)
