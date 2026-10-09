"""Train a reproducible, group-separated temporal pose decision tree.

Input can be either a directory of ``.npz`` clips or a manifest CSV with the
columns ``clip_path,label,group``.  Directory input uses this convention when
possible::

    clips/<label>/<source-group>/<clip>.npz

For a shallower layout, the immediate parent of a clip is treated as its
source group and the first directory below the input root as its label.  Each
NPZ must contain a numeric array shaped ``(frames, 33, 3)`` (``(33, 3)`` for a
single frame is also accepted); common keys such as ``landmarks`` and
``keypoints`` are recognized.  Arrays are always loaded with
``allow_pickle=False``.

Example::

    python train_pose_v2.py clips --output models/exercise_pose_v2.joblib
    python train_pose_v2.py clips.csv --seed 17

The output model is a standalone ``DecisionTreeClassifier``.  Metrics are
written next to it as ``exercise_pose_v2_metrics.json`` unless ``--metrics``
is supplied.  No equipment-detection target is created or trained here.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from pose_features_v2 import FEATURE_SIZE, extract_temporal_features


BENCHMARK_LABELS = (
    "bench_press",
    "shoulder_press",
    "push_up",
    "squat",
    "deadlift",
    "bicep_curl",
    "pull_up",
    "row",
    "lunge",
    "jumping_jack",
    "unknown",
)
NPZ_KEYS = ("landmarks", "poses", "pose", "xyz", "keypoints", "data", "arr_0")


@dataclass(frozen=True)
class ClipRecord:
    """One labeled clip and the source group that must stay in one split."""

    path: Path
    label: str
    group: str


def _clean_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _clip_array(array: np.ndarray, path: Path) -> np.ndarray:
    """Convert common clip layouts to ``(frames, 33, 3)`` without object data."""

    values = np.asarray(array)
    if values.dtype.kind in {"O", "V"}:
        raise ValueError(f"{path} contains an object/structured pose array.")
    if values.ndim == 2 and values.shape == (33, 3):
        return values[np.newaxis, ...].astype(np.float64, copy=False)
    if values.ndim == 3 and values.shape[1] == 33 and values.shape[2] >= 3:
        return values[:, :, :3].astype(np.float64, copy=False)
    if values.ndim == 2 and values.shape[1] == 99:
        return values.reshape(values.shape[0], 33, 3).astype(np.float64, copy=False)
    raise ValueError(
        f"{path} has unsupported pose shape {tuple(values.shape)}; "
        "expected (frames, 33, 3)."
    )


def load_clip_features(path: Path) -> np.ndarray:
    """Load one NPZ clip with pickle disabled and extract temporal features."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            selected = None
            for key in NPZ_KEYS:
                if key in archive.files:
                    selected = archive[key]
                    break
            if selected is None:
                numeric = [archive[key] for key in archive.files if archive[key].dtype.kind != "O"]
                if len(numeric) == 1:
                    selected = numeric[0]
            if selected is None:
                raise ValueError(
                    f"{path} has no pose array. Expected one of: {', '.join(NPZ_KEYS)}."
                )
            clip = _clip_array(selected, path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load pose clip {path}: {exc}") from exc
    return extract_temporal_features(clip)


def records_from_manifest(manifest_path: Path) -> list[ClipRecord]:
    """Read and validate the required ``clip_path,label,group`` CSV columns."""

    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"clip_path", "label", "group"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            found = ", ".join(reader.fieldnames or ())
            raise ValueError(f"Manifest must contain clip_path,label,group; found: {found}")
        records: list[ClipRecord] = []
        for line_number, row in enumerate(reader, start=2):
            raw_path = (row.get("clip_path") or "").strip()
            label = _clean_name(row.get("label") or "")
            group = (row.get("group") or "").strip()
            if not raw_path or not label or not group:
                raise ValueError(f"Manifest row {line_number} has an empty clip_path, label, or group.")
            clip_path = Path(raw_path)
            if not clip_path.is_absolute():
                clip_path = manifest_path.parent / clip_path
            records.append(ClipRecord(clip_path.resolve(), label, group))
    if not records:
        raise ValueError("Manifest contains no clips.")
    return records


def _infer_directory_record(root: Path, path: Path) -> ClipRecord:
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) < 2:
        raise ValueError(
            f"Cannot infer label/group for {path}; use <label>/<group>/<clip>.npz "
            "or provide a manifest."
        )
    label = _clean_name(parts[0])
    if len(parts) >= 3:
        group = "/".join(parts[1:-1])
    else:
        # A <label>/<group>.npz layout has an explicit source group in the
        # filename; this remains deterministic and avoids using clip names as
        # groups when multiple clips share a source.
        group = Path(parts[-1]).stem
    return ClipRecord(path.resolve(), label, group)


def records_from_directory(root: Path) -> list[ClipRecord]:
    """Discover NPZ clips and infer labels/groups from their relative paths."""

    if not root.is_dir():
        raise ValueError(f"Clip directory does not exist: {root}")
    paths = sorted(path for path in root.rglob("*.npz") if path.is_file())
    if not paths:
        raise ValueError(f"Clip directory contains no .npz files: {root}")
    records = [_infer_directory_record(root, path) for path in paths]
    return records


def load_records(source: Path) -> list[ClipRecord]:
    """Load records from a CSV manifest or a directory of NPZ clips."""

    if source.is_file() and source.suffix.lower() == ".csv":
        return records_from_manifest(source)
    if source.is_dir():
        return records_from_directory(source)
    raise ValueError(f"Input must be an existing directory or CSV manifest: {source}")


def _coverage(labels: Iterable[str]) -> dict[str, object]:
    values = list(labels)
    present = sorted(set(values))
    counts = {label: 0 for label in BENCHMARK_LABELS}
    for label in values:
        if label in counts:
            counts[label] += 1
    return {
        "present": present,
        "missing": [label for label in BENCHMARK_LABELS if label not in present],
        "benchmark_count": len(BENCHMARK_LABELS),
        "present_count": sum(label in present for label in BENCHMARK_LABELS),
        "benchmark_counts": counts,
    }


def _class_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, object]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(BENCHMARK_LABELS),
        target_names=list(BENCHMARK_LABELS),
        output_dict=True,
        zero_division=0,
    )
    per_class = {label: report[label] for label in BENCHMARK_LABELS}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
    }


def split_grouped_indices(
    records: Sequence[ClipRecord],
    seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> dict[str, np.ndarray]:
    """Split records by source group into disjoint train/validation/test indices.

    At least three distinct groups are required.  The deterministic search
    chooses a split close to the requested clip proportions while preferring a
    training split that contains every observed label.
    """

    if not records:
        raise ValueError("Cannot split an empty clip collection.")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction and validation_fraction must leave a positive test fraction.")
    groups = np.asarray([record.group for record in records], dtype=object)
    unique_groups = np.asarray(sorted(set(groups.tolist())), dtype=object)
    if unique_groups.size < 3:
        raise ValueError(
            f"Need at least 3 distinct source groups for train/validation/test; found {unique_groups.size}."
        )
    labels = np.asarray([record.label for record in records], dtype=object)
    desired = np.asarray([train_fraction, validation_fraction, 1.0 - train_fraction - validation_fraction])
    rng = np.random.default_rng(seed)
    best: tuple[float, dict[str, np.ndarray]] | None = None
    # Enumerating assignments for small group counts gives stable, high-quality
    # splits; randomized candidates cover larger datasets without exponential
    # work. Each candidate always assigns a whole group to exactly one split.
    attempts = 4096 if unique_groups.size <= 10 else 512
    for _ in range(attempts):
        order = rng.permutation(unique_groups)
        group_order = {group: position for position, group in enumerate(order)}
        assignment = rng.integers(0, 3, size=unique_groups.size)
        counts = np.bincount(assignment, minlength=3)
        for split_number in range(3):
            if counts[split_number] == 0:
                donor = int(np.argmax(counts))
                donor_positions = np.flatnonzero(assignment == donor)
                moved = int(donor_positions[0])
                assignment[moved] = split_number
                counts = np.bincount(assignment, minlength=3)
        split_indices = {
            name: np.asarray(
                [index for index, group in enumerate(groups) if assignment[group_order[group]] == split_number],
                dtype=int,
            )
            for split_number, name in enumerate(("train", "validation", "test"))
        }
        train_labels = set(labels[split_indices["train"]].tolist())
        missing_train = len(set(labels.tolist()) - train_labels)
        actual = np.asarray([len(split_indices[name]) for name in ("train", "validation", "test")], dtype=float)
        actual /= len(records)
        score = float(np.sum(np.abs(actual - desired)) + missing_train * 10.0)
        # Prefer candidates that do not put all examples of a label outside
        # validation/test when alternatives have the same size score.
        if best is None or score < best[0]:
            best = (score, split_indices)
            if score == 0:
                break
    if best is None:  # defensive; the minimum group check makes this unreachable
        raise ValueError("Unable to construct a grouped split.")
    result = best[1]
    if any(result[name].size == 0 for name in ("train", "validation", "test")):
        raise ValueError("Grouped split produced an empty partition; provide more source groups.")
    return result


def train_pipeline(
    source: Path,
    output_path: Path,
    metrics_path: Path | None = None,
    seed: int = 42,
    max_depth: int | None = 12,
    min_samples_leaf: int = 1,
) -> dict[str, object]:
    """Train the model, write separate model/metrics artifacts, and return metrics."""

    records = load_records(source)
    split = split_grouped_indices(records, seed=seed)
    features = np.vstack([load_clip_features(record.path) for record in records])
    labels = np.asarray([record.label for record in records], dtype=str)
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    model.fit(features[split["train"]], labels[split["train"]])
    split_metrics: dict[str, object] = {}
    for name in ("train", "validation", "test"):
        indices = split[name]
        predicted = model.predict(features[indices])
        split_metrics[name] = {
            "clips": int(indices.size),
            "groups": sorted({records[index].group for index in indices.tolist()}),
            "class_coverage": _coverage(labels[indices].tolist()),
            "metrics": _class_metrics(labels[indices].tolist(), predicted.tolist()),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    if metrics_path is None:
        metrics_path = output_path.with_name("exercise_pose_v2_metrics.json")
    metrics = {
        "model_type": "decision_tree",
        "feature_type": "normalized_mediapipe_33_xyz_temporal_summary",
        "feature_size": FEATURE_SIZE,
        "seed": seed,
        "source": str(source),
        "clips": len(records),
        "groups": len({record.group for record in records}),
        "benchmark_labels": list(BENCHMARK_LABELS),
        "class_coverage": _coverage(labels.tolist()),
        "splits": split_metrics,
        "equipment_detection": "not trained",
        "production_ready": False,
        "evaluation_warning": "Group disjointness is only as strong as supplied group IDs. clip_only_ groups do not establish source-video or subject independence.",
        "grouping_level": "clip_only" if any(r.group.startswith("clip_only_") for r in records) else "supplied_source_groups",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> None:
    """Parse the documented CLI and train the v2 pose model."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Clip directory or manifest CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("models") / "exercise_pose_v2.joblib",
        help="Standalone DecisionTreeClassifier output path.",
    )
    parser.add_argument("--metrics", type=Path, help="Metrics JSON output path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouped splitting/tree training.")
    parser.add_argument("--max-depth", type=int, default=12, help="Decision-tree max depth; use 0 for unlimited.")
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    args = parser.parse_args()
    max_depth = None if args.max_depth == 0 else args.max_depth
    metrics = train_pipeline(
        args.source,
        args.output,
        metrics_path=args.metrics,
        seed=args.seed,
        max_depth=max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    print(f"Trained on {metrics['clips']} clips across {metrics['groups']} groups.")
    print(f"Saved model to {args.output}")
    print(f"Saved metrics to {args.metrics or args.output.with_name('exercise_pose_v2_metrics.json')}")


if __name__ == "__main__":
    main()
