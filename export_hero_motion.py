"""Export real MediaPipe 3D pose sequences for the web hero animation.

Runs the same pose landmarker the app uses over frame sequences from
JIM_DATA29 and writes metric world landmarks (hip-centred, metres) to
web/public/motion/<name>.json, so the landing page replays captured
movement instead of a hand-animated figure. It also writes the 2D landmarks
and joint angles detected in the page photos to web/public/media/poses.json.

    .venv\\Scripts\\python.exe export_hero_motion.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from repetition_counter_v3 import visible_angles

ROOT = Path(__file__).parent
CLIPS = {
    "squat": ROOT / "JIM_DATA29" / "Squats" / "squats data29" / "images" / "train",
    "deadlift": ROOT / "JIM_DATA29" / "Deadlifts" / "deadlift Data29" / "image" / "Train",
}


def smooth(frames: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving average over time; damps landmark jitter."""
    kernel = np.ones(window) / window
    padded = np.pad(frames, ((window // 2, window // 2), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(frames)
    for joint in range(frames.shape[1]):
        for axis in range(3):
            out[:, joint, axis] = np.convolve(padded[:, joint, axis], kernel, mode="valid")
    return out


def export(name: str, folder: Path, stride: int, out_dir: Path, detector) -> None:
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")))[::stride]
    frames, last = [], None
    for path in files:
        image = cv2.resize(cv2.imread(path), (640, 360))
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        if result.pose_world_landmarks:
            last = [[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]]
        if last is not None:
            frames.append(last)
    data = smooth(np.asarray(frames, dtype=np.float64))
    payload = {
        "name": name,
        "fps": round(30 / stride, 2),
        "source": str(folder.relative_to(ROOT)).replace("\\", "/"),
        "frames": np.round(data, 3).tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.json"
    target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"{name}: {len(frames)} frames -> {target} ({target.stat().st_size // 1024} KB)")


def export_photo_poses(media_dir: Path, detector) -> None:
    poses = {}
    for path in sorted(media_dir.glob("*.jpg")):
        image = cv2.imread(str(path))
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        if not result.pose_landmarks:
            continue
        pose = result.pose_landmarks[0]
        poses[path.name] = {
            "landmarks": [[round(lm.x, 4), round(lm.y, 4), round(lm.visibility, 3)] for lm in pose],
            "angles": visible_angles(pose, 0.35),
        }
    target = media_dir / "poses.json"
    target.write_text(json.dumps(poses, separators=(",", ":")), encoding="utf-8")
    print(f"photo poses: {len(poses)} images -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=2, help="Use every Nth source frame (source is ~30 fps)")
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "motion")
    parser.add_argument("--media", type=Path, default=ROOT / "web" / "public" / "media")
    args = parser.parse_args()
    detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models" / "pose_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE,
    ))
    try:
        for name, folder in CLIPS.items():
            export(name, folder, args.stride, args.out, detector)
        export_photo_poses(args.media, detector)
    finally:
        detector.close()


if __name__ == "__main__":
    main()
