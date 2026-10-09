"""Experimental, visibility-gated pose-angle rep heuristics for v3 movements.

These counters are not validated measurements. A rep requires home -> effort ->
home with at least 0.4 seconds between the first home and completed return.
"""

from __future__ import annotations

import math
from typing import Optional


RULES = {
    "bench_press": ("elbow", 155, 100),
    "push_up": ("elbow", 155, 100),
    "shoulder_press": ("elbow", 155, 100),
    "bicep_curl": ("elbow", 150, 65),
    "pull_up": ("elbow", 150, 65),
    "row": ("elbow", 150, 100),
    "squat": ("squat", 160, 100),
    "deadlift": ("hip", 160, 100),
    "lunge": ("knee", 160, 100),
    "jumping_jack": ("shoulder", 35, 100),
}

JOINTS = {
    "elbow": (11, 13, 15),
    "shoulder": (23, 11, 13),
    "hip": (11, 23, 25),
    "knee": (23, 25, 27),
}


def _angle(landmarks, indices: tuple[int, int, int], visibility_threshold: float) -> Optional[float]:
    points = [landmarks[index] for index in indices]
    if any(
        not all(math.isfinite(float(getattr(point, axis))) for axis in ("x", "y"))
        or not math.isfinite(float(getattr(point, "visibility", 0)))
        or float(getattr(point, "visibility", 0)) < visibility_threshold
        for point in points
    ):
        return None
    a, b, c = points
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    norm = math.hypot(*ab) * math.hypot(*cb)
    if norm <= 1e-10:
        return None
    return round(math.degrees(math.acos(max(-1.0, min(1.0, (ab[0]*cb[0] + ab[1]*cb[1]) / norm)))), 1)


def visible_angles(landmarks, visibility_threshold: float = 0.35) -> dict[str, float]:
    """Only return angles whose complete joint triplet is visible and valid."""
    if landmarks is None or len(landmarks) < 33:
        return {}
    result = {}
    for side, offset in (("left", 0), ("right", 1)):
        for joint, indices in JOINTS.items():
            value = _angle(landmarks, tuple(index + offset for index in indices), visibility_threshold)
            if value is not None:
                result[f"{side}_{joint}"] = value
    return result


class RepetitionCounterV3:
    """One active-action cycle; clear stages when action or tracking changes."""

    def __init__(self, actions):
        self.counters = {action: 0 for action in actions if action != "unknown"}
        self.stages = {action: None for action in self.counters}
        self.active_action = "unknown"
        self.started_at: Optional[float] = None

    def clear_cycle(self) -> None:
        self.stages = dict.fromkeys(self.stages)
        self.active_action = "unknown"
        self.started_at = None

    def reset(self) -> dict[str, int]:
        self.counters = dict.fromkeys(self.counters, 0)
        self.clear_cycle()
        return self.counters.copy()

    @staticmethod
    def _position(action: str, angles: dict[str, float]) -> Optional[str]:
        joint, home_threshold, effort_threshold = RULES[action]
        if joint == "squat":
            pairs = [(angles[f"{side}_knee"], angles[f"{side}_hip"])
                     for side in ("left", "right")
                     if f"{side}_knee" in angles and f"{side}_hip" in angles]
            if not pairs:
                return None
            if all(knee >= home_threshold and hip >= home_threshold for knee, hip in pairs):
                return "home"
            if any(knee <= effort_threshold and hip <= effort_threshold for knee, hip in pairs):
                return "effort"
            return "middle"
        values = [angles[f"{side}_{joint}"] for side in ("left", "right")
                  if f"{side}_{joint}" in angles]
        if not values:
            return None
        if joint == "shoulder":
            if all(value < home_threshold for value in values):
                return "home"
            if any(value > effort_threshold for value in values):
                return "effort"
        else:
            if all(value >= home_threshold for value in values):
                return "home"
            if any(value <= effort_threshold for value in values):
                return "effort"
        return "middle"

    def update(self, action: str, angles: dict[str, float], timestamp: float) -> Optional[str]:
        if action != self.active_action:
            self.clear_cycle()
            self.active_action = action
        if action not in RULES:
            return None
        position = self._position(action, angles)
        if position is None:
            self.clear_cycle()
            return None
        stage = self.stages[action]
        if stage is None:
            if position == "home":
                self.stages[action] = "home"
                self.started_at = timestamp
        elif stage == "home" and position == "effort":
            self.stages[action] = "effort"
        elif stage == "effort" and position == "home":
            self.stages[action] = "home"
            start = self.started_at
            self.started_at = timestamp
            if start is not None and timestamp - start >= 0.4:
                self.counters[action] += 1
                return action
        return None
