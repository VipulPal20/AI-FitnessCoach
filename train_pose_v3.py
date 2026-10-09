"""Train a bounded hierarchical movement tree from grouped, timestamped NPZ clips.

CSV: clip_path,label,group (paths relative to CSV; group is a literal recording ID).
NPZ: landmarks (frames,33,3), timestamps (frames,) in seconds. Pickle is never
enabled for input. Only load a saved joblib artifact from a trusted local run.

Example: python train_pose_v3.py clips.csv --output models/exercise_pose_v3.joblib
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier

from movement_schema import ACTIONS, ACTION_FAMILY, FAMILY_MAP
from pose_features_v3 import (
    FEATURE_SIZE, MAX_GAP_SECONDS, SAMPLING_FPS, TIME_EPSILON, WINDOW_SECONDS,
    _validated_window, extract_window_features,
)

SPLITS = ("train", "validation", "test")
DEPTHS = (4, 6, 8, 10)
MIN_LEAVES = (2, 4, 8)
THRESHOLD = 0.7
MARGIN = 0.15


@dataclass(frozen=True)
class Clip:
    path: Path
    label: str
    group: str


def read_manifest(path: Path) -> list[Clip]:
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"clip_path", "label", "group"}.issubset(reader.fieldnames):
            raise ValueError("Manifest requires clip_path,label,group columns")
        clips: list[Clip] = []
        seen: set[Path] = set()
        for line, row in enumerate(reader, 2):
            raw = row.get("clip_path") or ""
            label = row.get("label") or ""
            group = row.get("group") or ""
            if not raw.strip() or not group.strip():
                raise ValueError(f"Row {line}: clip_path and group must be nonempty")
            if label not in ACTIONS:
                raise ValueError(f"Row {line}: invalid canonical label {label!r}")
            clip_path = (path.parent / raw).resolve()
            if clip_path in seen:
                raise ValueError(f"Row {line}: duplicate clip path {clip_path}")
            seen.add(clip_path)
            clips.append(Clip(clip_path, label, group))
    if not clips:
        raise ValueError("Manifest contains no clips")
    return clips


def clip_windows(path: Path) -> np.ndarray:
    """Produce windows every second, restarting at each continuous segment.

    A segment begins at its earliest timestamp and ends before a gap >0.35s.
    A window requires samples covering both its exact start and end. For
    floating-point NPZ timestamps, boundary poses are interpolated only within
    a continuous segment; the feature extractor receives exact endpoints.
    """

    with np.load(path, allow_pickle=False) as archive:
        if "landmarks" not in archive or "timestamps" not in archive:
            raise ValueError(f"{path}: NPZ requires landmarks and timestamps")
        poses, times = _validated_window(archive["landmarks"], archive["timestamps"])
    if times[-1] - times[0] < WINDOW_SECONDS - TIME_EPSILON:
        return np.empty((0, FEATURE_SIZE), dtype=np.float64)
    edges = np.r_[0, np.flatnonzero(np.diff(times) > MAX_GAP_SECONDS + 1e-9) + 1, len(times)]
    windows = []
    for begin, end in zip(edges[:-1], edges[1:]):
        t, p = times[begin:end], poses[begin:end]
        if len(t) < 2:
            continue
        count = int(np.floor((t[-1] - t[0] - WINDOW_SECONDS + TIME_EPSILON) / 1.0)) + 1
        for step in range(max(0, count)):
            start = t[0] + step
            finish = start + WINDOW_SECONDS
            indices = np.flatnonzero((t > start) & (t < finish))
            # Only the two boundary frames may be synthesized; never cross gaps.
            flat = p.reshape(len(p), -1)
            start_pose = np.array([np.interp(start, t, flat[:, j]) for j in range(99)]).reshape(33, 3)
            end_pose = np.array([np.interp(finish, t, flat[:, j]) for j in range(99)]).reshape(33, 3)
            window_poses = np.concatenate((start_pose[None], p[indices], end_pose[None]))
            window_times = np.r_[start, t[indices], finish]
            windows.append(extract_window_features(window_poses, window_times))
    return np.vstack(windows) if windows else np.empty((0, FEATURE_SIZE), dtype=np.float64)


def split_groups(groups: Sequence[str], labels: Sequence[str], seed: int = 42) -> dict[str, list[str]]:
    """Find a deterministic, class-covered 3-way recording-group split.

    Multi-label groups are indivisible. A bounded randomized greedy search
    selects coverage before matching the approximate 70/15/15 group sizes.
    Failure is explicit rather than silently evaluating a missing class.
    """

    by_group: dict[str, set[str]] = defaultdict(set)
    for group, label in zip(groups, labels):
        by_group[group].add(label)
    missing = [label for label in ACTIONS if not any(label in ls for ls in by_group.values())]
    if missing:
        raise ValueError(f"Missing required classes: {missing}")
    inadequate = {label: sum(label in ls for ls in by_group.values()) for label in ACTIONS}
    inadequate = {label: count for label, count in inadequate.items() if count < 3}
    if inadequate:
        raise ValueError(f"Each class needs >=3 independent recording group IDs: {inadequate}")
    rng = np.random.default_rng(seed)
    names = sorted(by_group)
    desired = np.array([0.7, 0.15, 0.15]) * len(names)
    best: tuple[float, dict[str, list[str]]] | None = None
    for _ in range(1024):
        assigned: dict[str, int] = {}
        covered = [set() for _ in SPLITS]
        tie = {name: float(rng.random()) for name in names}
        # Constrained classes first; this also distributes the mandatory
        # three groups of a rare class across all three partitions.
        for label in sorted(ACTIONS, key=lambda name: (sum(name in ls for ls in by_group.values()), name)):
            for slot in sorted(range(3), key=lambda i: (label in covered[i], i)):
                if label in covered[slot]:
                    continue
                candidates = [g for g in names if g not in assigned and label in by_group[g]]
                candidates.sort(key=lambda g: (
                    -sum(other not in covered[slot] for other in by_group[g]),
                    len([a for a in assigned.values() if a == slot]) / desired[slot],
                    tie[g],
                ))
                for group in candidates:
                    # Reserve enough still-unassigned groups to fill each
                    # outstanding split for every label this group carries.
                    possible = True
                    for other in by_group[group]:
                        remaining = sum(other in by_group[g] for g in names if g not in assigned and g != group)
                        outstanding = sum(other not in covered[i] for i in range(3) if i != slot)
                        if remaining < outstanding:
                            possible = False
                            break
                    if possible:
                        assigned[group] = slot
                        covered[slot].update(by_group[group])
                        break
                if label not in covered[slot]:
                    break
        if not all(set(ACTIONS) <= subset for subset in covered):
            continue
        for group in sorted((g for g in names if g not in assigned), key=lambda g: tie[g]):
            assigned[group] = min(range(3), key=lambda i: (sum(v == i for v in assigned.values()) / desired[i], i))
        result = {name: sorted(g for g, slot in assigned.items() if slot == i) for i, name in enumerate(SPLITS)}
        counts = np.array([len(result[name]) for name in SPLITS])
        score = float(np.abs(counts - desired).sum())
        if best is None or score < best[0]:
            best = (score, result)
    if best is None:
        raise ValueError("Cannot find a three-way class-covered group split; check overlapping group labels")
    return best[1]


def _fit(x: np.ndarray, y: np.ndarray, depth: int, leaf: int, seed: int) -> dict[str, Any]:
    families = np.array([ACTION_FAMILY[label] for label in y])
    def tree(target: np.ndarray, inputs: np.ndarray) -> DecisionTreeClassifier:
        model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=seed)
        model.fit(inputs, target)
        return model
    family_model = tree(families, x)
    conditional = {}
    for family, actions in FAMILY_MAP.items():
        if len(actions) > 1:
            selected = families == family
            conditional[family] = tree(y[selected], x[selected])
    return {
        "schema_version": 3,
        "model_type": "hierarchical_decision_tree",
        "feature_type": "normalized_mediapipe_temporal_20",
        "feature_size": FEATURE_SIZE,
        "window_seconds": WINDOW_SECONDS,
        "sampling_fps": SAMPLING_FPS,
        "labels": ACTIONS,
        "family_model": family_model,
        "exercise_models": conditional,
        "family_map": FAMILY_MAP,
        "threshold": THRESHOLD,
        "margin": MARGIN,
        "equipment_detection": "not_trained",
        "production_ready": False,
    }


def predict_hierarchy(artifact: dict[str, Any], features: np.ndarray) -> dict[str, float]:
    """Return all 11 joint family × conditional movement probabilities."""

    row = np.asarray(features, dtype=np.float64)
    if row.shape != (FEATURE_SIZE,) or not np.isfinite(row).all():
        raise ValueError(f"features must be a finite ({FEATURE_SIZE},) vector")
    row = row.reshape(1, -1)
    scores = {action: 0.0 for action in ACTIONS}
    family_model = artifact["family_model"]
    for family, family_probability in zip(family_model.classes_, family_model.predict_proba(row)[0]):
        actions = artifact["family_map"][family]
        if len(actions) == 1:
            scores[actions[0]] += float(family_probability)
        else:
            child = artifact["exercise_models"][family]
            for action, conditional_probability in zip(child.classes_, child.predict_proba(row)[0]):
                scores[action] += float(family_probability * conditional_probability)
    return scores


def _predict(artifact: dict[str, Any], x: np.ndarray) -> tuple[list[str], list[str]]:
    raw, abstained = [], []
    for row in x:
        probabilities = predict_hierarchy(artifact, row)
        ordered = sorted(ACTIONS, key=lambda action: -probabilities[action])
        top, runner_up = ordered[:2]
        raw.append(top)
        abstained.append(top if probabilities[top] >= artifact["threshold"] and
                         probabilities[top] - probabilities[runner_up] >= artifact["margin"]
                         else "unknown")
    return raw, abstained


def _report(y: Sequence[str], predictions: Sequence[str]) -> dict[str, Any]:
    report = classification_report(y, predictions, labels=ACTIONS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, labels=ACTIONS, average="macro", zero_division=0)),
        "per_class": {label: report[label] for label in ACTIONS},
        "confusion_labels": list(ACTIONS),
        "confusion_matrix": confusion_matrix(y, predictions, labels=ACTIONS).tolist(),
    }


def train_pipeline(manifest: Path, output: Path, metrics_path: Path | None = None, seed: int = 42) -> dict[str, Any]:
    clips = read_manifest(manifest)
    x_parts, window_labels, window_groups = [], [], []
    skipped = []
    for clip in clips:
        windows = clip_windows(clip.path)
        if len(windows) == 0:
            skipped.append(str(clip.path))
            continue
        x_parts.append(windows)
        window_labels.extend([clip.label] * len(windows))
        window_groups.extend([clip.group] * len(windows))
    # Validate only groups with actual usable windows; a short clip cannot
    # establish training or evaluation support.
    partition = split_groups(window_groups, window_labels, seed)
    x = np.vstack(x_parts)
    y = np.asarray(window_labels)
    ids = np.asarray(window_groups)
    indices = {name: np.flatnonzero(np.isin(ids, partition[name])) for name in SPLITS}
    train_idx, val_idx = indices["train"], indices["validation"]
    best: tuple[float, int, int, dict[str, Any]] | None = None
    search = []
    for depth in DEPTHS:
        for leaf in MIN_LEAVES:
            artifact = _fit(x[train_idx], y[train_idx], depth, leaf, seed)
            predicted, _ = _predict(artifact, x[val_idx])
            score = float(f1_score(y[val_idx], predicted, labels=ACTIONS, average="macro", zero_division=0))
            search.append({"max_depth": depth, "min_samples_leaf": leaf, "validation_macro_f1": score})
            if best is None or score > best[0]:
                best = (score, depth, leaf, artifact)
    assert best is not None
    artifact = best[3]  # trained on train only; test is untouched until selection
    sections = {}
    for name in SPLITS:
        subset = indices[name]
        raw, abstained = _predict(artifact, x[subset])
        truth = y[subset].tolist()
        sections[name] = {
            "windows": len(subset),
            "groups": partition[name],
            "group_count": len(partition[name]),
            "class_support": {label: truth.count(label) for label in ACTIONS},
            "missing_classes": [label for label in ACTIONS if label not in truth],
            "unknown_support": truth.count("unknown"),
            "raw": _report(truth, raw),
            "abstained": _report(truth, abstained),
        }
    metrics = {
        "schema_version": 3, "seed": seed, "source": str(Path(manifest).resolve()),
        "clips": len(clips), "usable_clips": len(clips) - len(skipped), "skipped_short_clips": skipped,
        "windows": len(x), "source_group_ids": sorted(set(window_groups)),
        "labels": list(ACTIONS), "selection_metric": "validation_raw_macro_f1",
        "search": search, "selected": {"max_depth": best[1], "min_samples_leaf": best[2]},
        "splits": sections, "equipment_detection": "not_trained", "production_ready": False,
    }
    output = Path(output)
    metrics_path = Path(metrics_path) if metrics_path is not None else output.with_name(output.stem + "_metrics.json")
    if output.resolve() == metrics_path.resolve():
        raise ValueError("Model and metrics paths must be distinct")
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with clip_path,label,group")
    parser.add_argument("--output", type=Path, required=True, help="Local joblib artifact path")
    parser.add_argument("--metrics", type=Path, help="JSON report path (defaults beside artifact)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        metrics = train_pipeline(args.manifest, args.output, args.metrics, args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"model": str(args.output), "windows": metrics["windows"],
                      "selected": metrics["selected"], "metrics": str(args.metrics or args.output.with_name(args.output.stem + "_metrics.json"))}))


if __name__ == "__main__":
    main()
