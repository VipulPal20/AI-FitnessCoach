"""Build unified exercise database from multiple data sources.

Merges:
  1. exercises-dataset (GitHub) — 1,324 exercises with GIFs, instructions, categories
  2. workoutexercises-images (Kaggle) — additional labelled exercise images

Produces:
  exercise_registry.json — canonical exercise database for the API

Usage:
    python build_exercise_registry.py
    python build_exercise_registry.py --github-data exercises-dataset/data/exercises.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ── Rep-counting heuristics per exercise category ────────────────────────────
# Each entry defines which joints to track and the angle thresholds for
# stage transitions (up/down). This lets the engine count reps for ANY
# exercise without hard-coded per-exercise logic.
#
# joint: MediaPipe landmark indices  (a, b, c) → angle at vertex b
# stage_down_below: angle must drop below this to enter "down" stage
# stage_up_above:   angle must rise above this to enter "up" stage
# rep_on:           which stage transition counts as a rep ("up" or "down")

REP_HEURISTICS: dict[str, dict[str, Any]] = {
    # ── Upper body push ──
    "bicep_curl": {
        "joints": {"primary": "elbow", "indices": [[11, 13, 15], [12, 14, 16]]},
        "stage_down_below": 35,
        "stage_up_above": 140,
        "rep_on": "up",
        "calories_per_rep": 0.3,
    },
    "overhead_press": {
        "joints": {"primary": "elbow", "indices": [[11, 13, 15], [12, 14, 16]]},
        "stage_down_below": 60,
        "stage_up_above": 130,
        "rep_on": "down",
        "calories_per_rep": 0.5,
    },
    "push_up": {
        "joints": {"primary": "elbow", "indices": [[5, 7, 9], [6, 8, 10]]},
        "stage_down_below": 90,
        "stage_up_above": 160,
        "rep_on": "up",
        "calories_per_rep": 0.4,
    },
    "bench_press": {
        "joints": {"primary": "elbow", "indices": [[11, 13, 15], [12, 14, 16]]},
        "stage_down_below": 80,
        "stage_up_above": 155,
        "rep_on": "up",
        "calories_per_rep": 0.6,
    },
    "dip": {
        "joints": {"primary": "elbow", "indices": [[11, 13, 15], [12, 14, 16]]},
        "stage_down_below": 80,
        "stage_up_above": 160,
        "rep_on": "up",
        "calories_per_rep": 0.5,
    },
    "lateral_raise": {
        "joints": {"primary": "shoulder", "indices": [[13, 11, 23], [14, 12, 24]]},
        "stage_down_below": 30,
        "stage_up_above": 75,
        "rep_on": "down",
        "calories_per_rep": 0.25,
    },
    "front_raise": {
        "joints": {"primary": "shoulder", "indices": [[13, 11, 23], [14, 12, 24]]},
        "stage_down_below": 30,
        "stage_up_above": 75,
        "rep_on": "down",
        "calories_per_rep": 0.25,
    },
    "tricep_extension": {
        "joints": {"primary": "elbow", "indices": [[11, 13, 15], [12, 14, 16]]},
        "stage_down_below": 50,
        "stage_up_above": 150,
        "rep_on": "up",
        "calories_per_rep": 0.3,
    },
    "cable_fly": {
        "joints": {"primary": "elbow", "indices": [[11, 13, 15], [12, 14, 16]]},
        "stage_down_below": 100,
        "stage_up_above": 160,
        "rep_on": "up",
        "calories_per_rep": 0.3,
    },
    # ── Upper body pull ──
    "pull_up": {
        "joints": {"primary": "elbow", "indices": [[11, 13, 15], [12, 14, 16]]},
        "stage_down_below": 70,
        "stage_up_above": 155,
        "rep_on": "down",
        "calories_per_rep": 1.0,
    },
    "lat_pulldown": {
        "joints": {"primary": "elbow", "indices": [[11, 13, 15], [12, 14, 16]]},
        "stage_down_below": 70,
        "stage_up_above": 155,
        "rep_on": "down",
        "calories_per_rep": 0.4,
    },
    "barbell_row": {
        "joints": {"primary": "elbow", "indices": [[11, 13, 15], [12, 14, 16]]},
        "stage_down_below": 60,
        "stage_up_above": 150,
        "rep_on": "down",
        "calories_per_rep": 0.5,
    },
    "cable_row": {
        "joints": {"primary": "elbow", "indices": [[11, 13, 15], [12, 14, 16]]},
        "stage_down_below": 60,
        "stage_up_above": 150,
        "rep_on": "down",
        "calories_per_rep": 0.4,
    },
    "face_pull": {
        "joints": {"primary": "elbow", "indices": [[11, 13, 15], [12, 14, 16]]},
        "stage_down_below": 60,
        "stage_up_above": 140,
        "rep_on": "down",
        "calories_per_rep": 0.25,
    },
    # ── Lower body ──
    "squat": {
        "joints": {"primary": "knee", "indices": [[23, 25, 27], [24, 26, 28]]},
        "stage_down_below": 100,
        "stage_up_above": 155,
        "rep_on": "up",
        "calories_per_rep": 0.5,
    },
    "lunge": {
        "joints": {"primary": "knee", "indices": [[23, 25, 27], [24, 26, 28]]},
        "stage_down_below": 100,
        "stage_up_above": 155,
        "rep_on": "up",
        "calories_per_rep": 0.4,
    },
    "deadlift": {
        "joints": {"primary": "hip", "indices": [[11, 23, 25], [12, 24, 26]]},
        "stage_down_below": 100,
        "stage_up_above": 160,
        "rep_on": "up",
        "calories_per_rep": 0.7,
    },
    "hip_thrust": {
        "joints": {"primary": "hip", "indices": [[11, 23, 25], [12, 24, 26]]},
        "stage_down_below": 100,
        "stage_up_above": 160,
        "rep_on": "up",
        "calories_per_rep": 0.5,
    },
    "leg_extension": {
        "joints": {"primary": "knee", "indices": [[23, 25, 27], [24, 26, 28]]},
        "stage_down_below": 80,
        "stage_up_above": 155,
        "rep_on": "up",
        "calories_per_rep": 0.3,
    },
    "leg_curl": {
        "joints": {"primary": "knee", "indices": [[23, 25, 27], [24, 26, 28]]},
        "stage_down_below": 60,
        "stage_up_above": 140,
        "rep_on": "down",
        "calories_per_rep": 0.3,
    },
    "calf_raise": {
        "joints": {"primary": "ankle", "indices": [[25, 27, 31], [26, 28, 32]]},
        "stage_down_below": 80,
        "stage_up_above": 110,
        "rep_on": "up",
        "calories_per_rep": 0.15,
    },
    # ── Core ──
    "russian_twist": {
        "joints": {"primary": "hip", "indices": [[11, 23, 25], [12, 24, 26]]},
        "stage_down_below": 70,
        "stage_up_above": 110,
        "rep_on": "up",
        "calories_per_rep": 0.2,
    },
    "crunch": {
        "joints": {"primary": "hip", "indices": [[11, 23, 25], [12, 24, 26]]},
        "stage_down_below": 60,
        "stage_up_above": 130,
        "rep_on": "down",
        "calories_per_rep": 0.15,
    },
    "sit_up": {
        "joints": {"primary": "hip", "indices": [[11, 23, 25], [12, 24, 26]]},
        "stage_down_below": 60,
        "stage_up_above": 140,
        "rep_on": "down",
        "calories_per_rep": 0.2,
    },
    "leg_raise": {
        "joints": {"primary": "hip", "indices": [[11, 23, 25], [12, 24, 26]]},
        "stage_down_below": 50,
        "stage_up_above": 140,
        "rep_on": "down",
        "calories_per_rep": 0.25,
    },
    # ── Cardio / Plyometric ──
    "jumping_jack": {
        "joints": {"primary": "shoulder", "indices": [[13, 11, 23], [14, 12, 24]]},
        "stage_down_below": 30,
        "stage_up_above": 100,
        "rep_on": "down",
        "calories_per_rep": 0.2,
    },
    "burpee": {
        "joints": {"primary": "hip", "indices": [[11, 23, 25], [12, 24, 26]]},
        "stage_down_below": 60,
        "stage_up_above": 155,
        "rep_on": "up",
        "calories_per_rep": 1.0,
    },
}

# Mapping from exercises-dataset names to canonical rep heuristic keys
NAME_TO_HEURISTIC: dict[str, str] = {
    # Curls
    "bicep curl": "bicep_curl", "biceps curl": "bicep_curl", "hammer curl": "bicep_curl",
    "preacher curl": "bicep_curl", "concentration curl": "bicep_curl",
    "barbell curl": "bicep_curl", "cable curl": "bicep_curl", "ez-bar curl": "bicep_curl",
    "dumbbell curl": "bicep_curl", "incline curl": "bicep_curl",
    "reverse curl": "bicep_curl", "spider curl": "bicep_curl",
    # Press
    "overhead press": "overhead_press", "shoulder press": "overhead_press",
    "military press": "overhead_press", "arnold press": "overhead_press",
    "barbell bench press": "bench_press", "bench press": "bench_press",
    "dumbbell bench press": "bench_press", "incline bench press": "bench_press",
    "decline bench press": "bench_press", "chest press": "bench_press",
    # Push-up
    "push-up": "push_up", "push up": "push_up", "archer push up": "push_up",
    "diamond push-up": "push_up", "wide push-up": "push_up",
    # Pull-up
    "pull-up": "pull_up", "pull up": "pull_up", "chin-up": "pull_up",
    "archer pull up": "pull_up",
    # Rows
    "barbell row": "barbell_row", "bent over row": "barbell_row",
    "cable row": "cable_row", "seated row": "cable_row",
    "dumbbell row": "barbell_row", "t-bar row": "barbell_row",
    # Lat pulldown
    "lat pulldown": "lat_pulldown", "pulldown": "lat_pulldown",
    # Raises
    "lateral raise": "lateral_raise", "side raise": "lateral_raise",
    "front raise": "front_raise",
    # Dip
    "dip": "dip", "chest dip": "dip", "tricep dip": "dip",
    # Tricep
    "tricep extension": "tricep_extension", "triceps extension": "tricep_extension",
    "skull crusher": "tricep_extension", "overhead extension": "tricep_extension",
    "tricep pushdown": "tricep_extension", "cable pushdown": "tricep_extension",
    # Fly
    "cable fly": "cable_fly", "chest fly": "cable_fly", "dumbbell fly": "cable_fly",
    "cable crossover": "cable_fly", "pec deck": "cable_fly",
    # Face pull
    "face pull": "face_pull",
    # Squat
    "squat": "squat", "full squat": "squat", "goblet squat": "squat",
    "front squat": "squat", "hack squat": "squat", "sumo squat": "squat",
    "barbell squat": "squat", "smith machine squat": "squat",
    "pistol squat": "squat", "split squat": "squat",
    "barbell full squat": "squat",
    # Lunge
    "lunge": "lunge", "walking lunge": "lunge", "reverse lunge": "lunge",
    "bulgarian split squat": "lunge",
    # Deadlift
    "deadlift": "deadlift", "barbell deadlift": "deadlift",
    "romanian deadlift": "deadlift", "sumo deadlift": "deadlift",
    "stiff-leg deadlift": "deadlift", "trap bar deadlift": "deadlift",
    # Hip thrust
    "hip thrust": "hip_thrust", "glute bridge": "hip_thrust",
    # Leg extension / curl
    "leg extension": "leg_extension", "leg curl": "leg_curl",
    # Calf raise
    "calf raise": "calf_raise", "standing calf raise": "calf_raise",
    "seated calf raise": "calf_raise",
    # Core
    "russian twist": "russian_twist", "crunch": "crunch", "sit-up": "sit_up",
    "sit up": "sit_up", "leg raise": "leg_raise", "hanging leg raise": "leg_raise",
    # Cardio
    "jumping jack": "jumping_jack", "jumping jacks": "jumping_jack",
    "burpee": "burpee", "burpees": "burpee",
}


def normalize_name(name: str) -> str:
    """Lowercase, strip parentheticals, collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def match_heuristic(exercise_name: str) -> str | None:
    """Try to match an exercise name to a rep-counting heuristic key."""
    name = normalize_name(exercise_name)
    # Direct match
    if name in NAME_TO_HEURISTIC:
        return NAME_TO_HEURISTIC[name]
    # Substring match (longest first)
    sorted_keys = sorted(NAME_TO_HEURISTIC.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in name:
            return NAME_TO_HEURISTIC[key]
    return None


def build_registry(github_json_path: Path, kaggle_path: Path | None, output_path: Path) -> dict:
    """Build the unified exercise registry JSON."""
    # ── Load GitHub exercises-dataset ──
    exercises: list[dict[str, Any]] = []
    if github_json_path.is_file():
        raw = json.loads(github_json_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(raw)} exercises from GitHub dataset")

        for entry in raw:
            name = entry.get("name", "")
            heuristic_key = match_heuristic(name)
            heuristic = REP_HEURISTICS.get(heuristic_key) if heuristic_key else None

            exercise = {
                "id": entry.get("id", ""),
                "name": name,
                "category": entry.get("category", ""),
                "body_part": entry.get("body_part", ""),
                "equipment": entry.get("equipment", ""),
                "target": entry.get("target", ""),
                "muscle_group": entry.get("muscle_group", ""),
                "secondary_muscles": entry.get("secondary_muscles", []),
                "instructions": entry.get("instructions", {}),
                "image": entry.get("image", ""),
                "gif_url": entry.get("gif_url", ""),
                "rep_counting": {
                    "supported": heuristic is not None,
                    "heuristic_key": heuristic_key,
                    "calories_per_rep": heuristic["calories_per_rep"] if heuristic else 0.0,
                },
            }
            exercises.append(exercise)
    else:
        print(f"Warning: GitHub dataset not found at {github_json_path}")

    # ── Merge Kaggle images (if available) ──
    kaggle_exercise_count = 0
    if kaggle_path and kaggle_path.is_dir():
        for category_dir in sorted(kaggle_path.iterdir()):
            if not category_dir.is_dir():
                continue
            for exercise_dir in sorted(category_dir.iterdir()):
                if not exercise_dir.is_dir():
                    continue
                images = list(exercise_dir.glob("*.jpg")) + list(exercise_dir.glob("*.png")) + list(exercise_dir.glob("*.jpeg"))
                if images:
                    kaggle_exercise_count += 1
        print(f"Found {kaggle_exercise_count} exercise folders in Kaggle dataset")

    # ── Build summary statistics ──
    rep_supported = sum(1 for e in exercises if e["rep_counting"]["supported"])
    categories = Counter(e["category"] for e in exercises)
    equipment_types = Counter(e["equipment"] for e in exercises)

    registry = {
        "version": "2.0.0",
        "total_exercises": len(exercises),
        "rep_counting_supported": rep_supported,
        "categories": dict(categories.most_common()),
        "equipment_types": dict(equipment_types.most_common()),
        "heuristics": REP_HEURISTICS,
        "exercises": exercises,
    }

    output_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRegistry saved to {output_path}")
    print(f"  Total exercises: {len(exercises)}")
    print(f"  Rep counting supported: {rep_supported}")
    print(f"  Categories: {len(categories)}")
    print(f"  Equipment types: {len(equipment_types)}")

    return registry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unified exercise registry.")
    parser.add_argument(
        "--github-data",
        type=Path,
        default=Path(__file__).with_name("exercises-dataset") / "data" / "exercises.json",
    )
    parser.add_argument(
        "--kaggle-data",
        type=Path,
        default=None,
        help="Path to Kaggle workoutexercises-images dataset root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("exercise_registry.json"),
    )
    args = parser.parse_args()
    build_registry(args.github_data, args.kaggle_data, args.output)
