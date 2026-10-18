"""Extract bounded, grouped MediaPipe Pose v2 clips from a video manifest.

The input CSV is intentionally explicit.  It must contain these columns::

    path,label,group,source_url,license

``path`` is resolved relative to the manifest when it is not absolute.  The
other metadata is not inferred: labels and source groups are copied exactly to
the generated clip manifest. A row produces clips only with finite 33-landmark
poses and at least one visible complete body side, matching the v3 runtime.
Missing pose frames terminate a sequence; frames are never interpolated or
turned into synthetic positives.

The generated ``manifest_pose_v2.csv`` has the contract consumed by
``merge_pose_manifests.py`` and ``train_pose_v3.py``::

    clip_path,label,group

Each NPZ contains ``landmarks`` shaped ``(frames, 33, 3)`` and ``timestamps``.
The XYZ values are MediaPipe's normalized landmark coordinates (float32).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from pose_features_v3 import MAX_GAP_SECONDS


LANDMARK_COUNT = 33
XYZ_DIMENSIONS = 3
DEFAULT_FPS = 10.0
DEFAULT_VISIBILITY_THRESHOLD = 0.35
DEFAULT_MIN_CLIP_SECONDS = 1.0
DEFAULT_MAX_CLIP_SECONDS = 10.0
DEFAULT_MAX_VIDEO_SECONDS = 300.0
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker.task"


@dataclass(frozen=True)
class ManifestRow:
    """One source video and its explicitly provided metadata."""

    row_number: int
    path: Path
    label: str
    group: str
    source_url: str
    license: str


@dataclass(frozen=True)
class ExtractConfig:
    """Bounds and pose quality settings for one extraction run."""

    fps: float = DEFAULT_FPS
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD
    min_clip_seconds: float = DEFAULT_MIN_CLIP_SECONDS
    max_clip_seconds: float = DEFAULT_MAX_CLIP_SECONDS
    max_video_seconds: float = DEFAULT_MAX_VIDEO_SECONDS
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    @property
    def min_clip_frames(self) -> int:
        return max(1, int(math.ceil(self.min_clip_seconds * self.fps)))

    @property
    def max_clip_frames(self) -> int:
        return max(self.min_clip_frames, int(math.floor(self.max_clip_seconds * self.fps)))


def _require_finite_range(name: str, value: float, low: float, high: float | None = None) -> None:
    if not math.isfinite(value) or value < low or (high is not None and value > high):
        limit = f" to {high}" if high is not None else ""
        raise ValueError(f"{name} must be finite and in [{low}{limit}], got {value!r}")


def validate_config(config: ExtractConfig) -> None:
    _require_finite_range("fps", config.fps, np.finfo(float).eps)
    _require_finite_range("visibility_threshold", config.visibility_threshold, 0.0, 1.0)
    _require_finite_range("min_clip_seconds", config.min_clip_seconds, np.finfo(float).eps)
    _require_finite_range("max_clip_seconds", config.max_clip_seconds, config.min_clip_seconds)
    # A video may intentionally be shorter than the per-clip ceiling; the
    # source bound only needs to allow the minimum useful clip duration.
    _require_finite_range("max_video_seconds", config.max_video_seconds, config.min_clip_seconds)
    for name, value in (
        ("min_detection_confidence", config.min_detection_confidence),
        ("min_presence_confidence", config.min_presence_confidence),
        ("min_tracking_confidence", config.min_tracking_confidence),
    ):
        _require_finite_range(name, value, 0.0, 1.0)


def _field_name(fieldnames: Sequence[str], preferred: str, aliases: Iterable[str] = ()) -> str:
    """Find a required field without silently guessing a value."""

    for candidate in (preferred, *aliases):
        if candidate in fieldnames:
            return candidate
    raise ValueError(f"Manifest is missing required column {preferred!r}.")


def read_video_manifest(manifest_path: Path) -> list[ManifestRow]:
    """Read the explicit video manifest and resolve only its listed paths."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Video manifest does not exist: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name.strip() for name in (reader.fieldnames or []) if name is not None]
        if not fieldnames:
            raise ValueError("Video manifest has no header row.")
        path_field = _field_name(fieldnames, "path", ("video_path",))
        for required in ("label", "group", "source_url", "license"):
            _field_name(fieldnames, required)

        rows: list[ManifestRow] = []
        for line_number, raw in enumerate(reader, start=2):
            # DictReader keys are the original header strings.  Read through a
            # small normalized mapping so surrounding whitespace in headers is
            # harmless, while values remain explicit and unmodified.
            row = {(key or "").strip(): (value or "").strip() for key, value in raw.items()}
            raw_path = row.get(path_field, "")
            label = row.get("label", "")
            group = row.get("group", "")
            if not raw_path:
                raise ValueError(f"Manifest row {line_number} has an empty path.")
            if not label:
                raise ValueError(f"Manifest row {line_number} has an empty label; no label is inferred.")
            if not group:
                raise ValueError(f"Manifest row {line_number} has an empty group; no group is inferred.")
            source_path = Path(raw_path).expanduser()
            if not source_path.is_absolute():
                source_path = manifest_path.parent / source_path
            rows.append(
                ManifestRow(
                    row_number=line_number,
                    path=source_path.resolve(),
                    label=label,
                    group=group,
                    source_url=row.get("source_url", ""),
                    license=row.get("license", ""),
                )
            )
    if not rows:
        raise ValueError("Video manifest contains no data rows.")
    return rows


def _landmarks_to_xyz(pose_landmarks: Any, visibility_threshold: float) -> np.ndarray | None:
    """Return one gated ``(33, 3)`` pose, or ``None`` for a missing frame."""

    from pose_features_v3 import landmarks_to_xyz
    return landmarks_to_xyz(pose_landmarks, visibility_threshold)


def _legacy_all_landmarks_to_xyz(pose_landmarks: Any, visibility_threshold: float) -> np.ndarray | None:
    if pose_landmarks is None or len(pose_landmarks) != LANDMARK_COUNT:
        return None
    values = np.empty((LANDMARK_COUNT, XYZ_DIMENSIONS), dtype=np.float32)
    for index, landmark in enumerate(pose_landmarks):
        visibility = getattr(landmark, "visibility", None)
        if visibility is None:
            return None
        try:
            visibility_value = float(visibility)
            xyz = (float(landmark.x), float(landmark.y), float(landmark.z))
        except (AttributeError, TypeError, ValueError):
            return None
        if not math.isfinite(visibility_value) or visibility_value < visibility_threshold:
            return None
        if not np.isfinite(xyz).all():
            return None
        values[index] = xyz
    return values


def pose_result_to_xyz(result: Any, visibility_threshold: float) -> np.ndarray | None:
    """Extract the first configured pose from a MediaPipe task result."""

    poses = getattr(result, "pose_landmarks", None)
    if not poses:
        return None
    return _landmarks_to_xyz(poses[0], visibility_threshold)


def _make_landmarker(model_path: Path, config: ExtractConfig) -> Any:
    """Create an IMAGE-mode task landmarker without downloading a model."""

    if not model_path.is_file():
        raise FileNotFoundError(f"Pose landmarker model not found: {model_path}")
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("mediapipe is required for extraction; no model download is attempted.") from exc

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        # Match the runtime pose detector; sampled timestamps are retained
        # for the temporal descriptor rather than MediaPipe tracking.
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=config.min_detection_confidence,
        min_pose_presence_confidence=config.min_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return mp, vision.PoseLandmarker.create_from_options(options)


def _save_clip(
    clip_path: Path,
    frames: Sequence[np.ndarray],
    timestamps: Sequence[float],
) -> None:
    values = np.asarray(frames, dtype=np.float32)
    times = np.asarray(timestamps, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (LANDMARK_COUNT, XYZ_DIMENSIONS):
        raise ValueError(f"Internal clip has invalid pose shape: {values.shape}")
    if values.shape[0] != times.shape[0] or values.shape[0] == 0:
        raise ValueError("Internal clip has inconsistent timestamps.")
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(clip_path, landmarks=values, timestamps=times)


def extract_video(
    row: ManifestRow,
    output_dir: Path,
    model_path: Path,
    config: ExtractConfig,
    mp_module: Any | None = None,
    detector: Any | None = None,
) -> list[Path]:
    """Extract one source row into bounded clips, splitting at missing poses.

    ``mp_module`` and ``detector`` are optional injection points for callers
    that already own a landmarker or for tests; normal CLI use creates them
    once per video and releases them after the call.
    """

    validate_config(config)
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("opencv-python is required for video extraction.") from exc
    if not row.path.is_file():
        raise FileNotFoundError(f"Video listed in manifest does not exist: {row.path}")

    owned_detector = detector is None
    if detector is None:
        mp_module, detector = _make_landmarker(model_path, config)
    if mp_module is None:
        raise ValueError("mp_module is required when detector is supplied.")

    capture = cv2.VideoCapture(str(row.path))
    if not capture.isOpened():
        if owned_detector and hasattr(detector, "close"):
            detector.close()
        raise RuntimeError(f"Could not open video: {row.path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = config.fps
    sample_interval = 1.0 / config.fps
    next_sample_time = 0.0
    last_timestamp_ms = -1
    frame_index = 0
    current_frames: list[np.ndarray] = []
    current_times: list[float] = []
    last_valid_timestamp: float | None = None
    clip_paths: list[Path] = []
    clip_number = 0

    def flush_current() -> None:
        nonlocal clip_number, current_frames, current_times, last_valid_timestamp
        if len(current_frames) < config.min_clip_frames:
            current_frames = []
            current_times = []
            last_valid_timestamp = None
            return
        safe_stem = f"row_{row.row_number:06d}"
        while current_frames:
            frames = current_frames[: config.max_clip_frames]
            times = current_times[: config.max_clip_frames]
            if len(frames) >= config.min_clip_frames:
                relative = Path("clips") / f"{safe_stem}_clip_{clip_number:04d}.npz"
                target = output_dir / relative
                _save_clip(target, frames, times)
                clip_paths.append(relative)
                clip_number += 1
            current_frames = current_frames[config.max_clip_frames :]
            current_times = current_times[config.max_clip_frames :]
            if not current_frames:
                last_valid_timestamp = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            frame_index += 1
            if timestamp > config.max_video_seconds:
                break
            if timestamp + (0.5 / source_fps) < next_sample_time:
                continue
            while next_sample_time <= timestamp:
                next_sample_time += sample_interval

            # MediaPipe Tasks expects RGB pixel data; no frame is synthesized
            # when decoding or pose detection fails.
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                factor = 640 / max(h, w)
                frame = cv2.resize(frame, (round(w * factor), round(h * factor)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
            timestamp_ms = max(last_timestamp_ms + 1, int(round(timestamp * 1000.0)))
            last_timestamp_ms = timestamp_ms
            # Match the live engine's per-image pose mode exactly. Temporal
            # modeling is in the shared descriptor, not a different tracker.
            result = detector.detect(image)
            xyz = pose_result_to_xyz(result, config.visibility_threshold)
            if xyz is None:
                # A short detector miss is represented by a timestamp gap and
                # remains usable if the v3 window gap contract is preserved.
                # Long gaps terminate the segment; no pose is synthesized.
                continue
            if last_valid_timestamp is not None and timestamp - last_valid_timestamp > MAX_GAP_SECONDS:
                flush_current()
            current_frames.append(xyz)
            current_times.append(timestamp)
            last_valid_timestamp = timestamp
            if len(current_frames) >= config.max_clip_frames:
                flush_current()
    finally:
        capture.release()
        if owned_detector and hasattr(detector, "close"):
            detector.close()
    flush_current()
    return clip_paths


def _write_clip_manifest(output_dir: Path, records: Sequence[tuple[Path, str, str]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest_pose_v2.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("clip_path", "label", "group"))
        for clip_path, label, group in records:
            writer.writerow((clip_path.as_posix(), label, group))
    return manifest_path


def _delete_manifest_sources(rows: Sequence[ManifestRow]) -> tuple[list[str], list[dict[str, str]]]:
    """Delete only listed regular files; never recurse or remove directories."""

    deleted: list[str] = []
    failures: list[dict[str, str]] = []
    seen: set[Path] = set()
    for row in rows:
        print(f"Extracting {row.row_number - 1}/{len(rows)}: {row.label}", flush=True)
        if row.path in seen:
            continue
        seen.add(row.path)
        try:
            if row.path.is_dir():
                failures.append({"path": str(row.path), "error": "listed path is a directory"})
            elif not row.path.exists() and not row.path.is_symlink():
                failures.append({"path": str(row.path), "error": "listed file no longer exists"})
            else:
                row.path.unlink()
                deleted.append(str(row.path))
        except OSError as exc:
            failures.append({"path": str(row.path), "error": str(exc)})
    return deleted, failures


def run_extraction(
    manifest_path: Path,
    output_dir: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    config: ExtractConfig = ExtractConfig(),
    delete_source_media: bool = False,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run extraction, write outputs, and return the receipt payload."""

    validate_config(config)
    rows = read_video_manifest(manifest_path)
    if labels:
        allowed = set(labels)
        rows = [row for row in rows if row.label in allowed]
        if not rows:
            raise ValueError(f"No manifest rows matched labels: {sorted(allowed)}")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_records: list[tuple[Path, str, str]] = []
    row_receipts: list[dict[str, Any]] = []
    for row in rows:
        try:
            paths = extract_video(row, output_dir, Path(model_path).expanduser().resolve(), config)
            if not paths:
                raise RuntimeError("no bounded clips met the pose visibility and duration gates")
            clip_records.extend((path, row.label, row.group) for path in paths)
            row_receipts.append(
                {
                    "row": row.row_number,
                    "path": str(row.path),
                    "label": row.label,
                    "group": row.group,
                    "source_url": row.source_url,
                    "license": row.license,
                    "status": "ok",
                    "clips": [str(p) for p in paths],
                }
            )
        except Exception as exc:  # keep independent manifest rows reportable
            row_receipts.append(
                {
                    "row": row.row_number,
                    "path": str(row.path),
                    "label": row.label,
                    "group": row.group,
                    "source_url": row.source_url,
                    "license": row.license,
                    "status": "failed",
                    "error": str(exc),
                    "clips": [],
                }
            )

    clip_manifest = _write_clip_manifest(output_dir, clip_records)
    failures = [receipt for receipt in row_receipts if receipt["status"] != "ok"]
    deleted: list[str] = []
    deletion_failures: list[dict[str, str]] = []
    if delete_source_media:
        # Deletion is intentionally all-or-nothing for a run: a failed row
        # means source evidence is retained for inspection and retry.
        if not failures and clip_records:
            deleted, deletion_failures = _delete_manifest_sources(rows)
        else:
            deletion_failures = [{"path": str(row.path), "error": "source retained because extraction was not fully successful"} for row in rows]

    receipt: dict[str, Any] = {
        "success": not failures and bool(clip_records) and not deletion_failures,
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "output_dir": str(output_dir),
        "clip_manifest": str(clip_manifest),
        "model": str(Path(model_path).expanduser().resolve()),
        "settings": asdict(config),
        "rows": row_receipts,
        "clip_count": len(clip_records),
        "delete_source_media_requested": delete_source_media,
        "deleted_source_media": deleted,
        "deletion_failures": deletion_failures,
    }
    receipt_path = output_dir / "extraction_receipt.json"
    with receipt_path.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    receipt["receipt"] = str(receipt_path)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract bounded MediaPipe Pose 33-landmark XYZ clips from an explicit video manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="CSV with path,label,group,source_url,license columns")
    parser.add_argument("output_dir", type=Path, help="Directory for NPZ clips, manifest_pose_v2.csv, and receipt")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Chronological sampling rate")
    parser.add_argument("--visibility-threshold", type=float, default=DEFAULT_VISIBILITY_THRESHOLD)
    parser.add_argument("--min-clip-seconds", type=float, default=DEFAULT_MIN_CLIP_SECONDS)
    parser.add_argument("--max-clip-seconds", type=float, default=DEFAULT_MAX_CLIP_SECONDS)
    parser.add_argument("--max-video-seconds", type=float, default=DEFAULT_MAX_VIDEO_SECONDS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Existing local pose_landmarker.task")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--delete-source-media",
        action="store_true",
        help="After a fully successful run, unlink only regular files listed in the manifest",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Restrict extraction to one or more exact labels; may be repeated",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = ExtractConfig(
        fps=args.fps,
        visibility_threshold=args.visibility_threshold,
        min_clip_seconds=args.min_clip_seconds,
        max_clip_seconds=args.max_clip_seconds,
        max_video_seconds=args.max_video_seconds,
        min_detection_confidence=args.min_detection_confidence,
        min_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    try:
        receipt = run_extraction(
            args.manifest,
            args.output_dir,
            model_path=args.model,
            config=config,
            delete_source_media=args.delete_source_media,
            labels=args.labels,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps({key: receipt[key] for key in ("success", "clip_count", "clip_manifest", "receipt", "deleted_source_media")}, indent=2))
    return 0 if receipt["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
