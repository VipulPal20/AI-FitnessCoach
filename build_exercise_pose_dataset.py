"""Normalize the supplied exercise pose archive into a YOLO-pose dataset."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = {"Deadlifts": 0, "shoulders": 1, "Squats": 2}


def split_from_path(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    if "train" in parts:
        return "train"
    if "val" in parts:
        return "val"
    return None


def matching_label(image_path: Path) -> Path | None:
    """Find a label with the same filename in the sibling label(s) directory."""
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part.lower() in {"image", "images"}:
            for label_dir in ("label", "labels"):
                candidate = Path(*parts[:index], label_dir, *parts[index + 1:]).with_suffix(".txt")
                if candidate.is_file():
                    return candidate
    return None


def rewrite_label(source: Path, destination: Path, class_id: int) -> None:
    rewritten: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        values = line.split()
        # YOLO pose: class + bbox(4) + 17 keypoints(x, y, visibility). A small
        # subset of the supplied labels omits trailing invisible keypoints;
        # restore them as (0, 0, 0) instead of losing otherwise useful frames.
        if len(values) < 5 or (len(values) - 5) % 3:
            raise ValueError(f"Malformed pose annotation in {source}: got {len(values)} values")
        if len(values) > 56:
            raise ValueError(f"Too many keypoints in {source}: got {len(values)} values")
        values.extend(["0", "0", "0"] * ((56 - len(values)) // 3))
        values[0] = str(class_id)
        rewritten.append(" ".join(values))
    destination.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")


def build(source: Path, destination: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)

    for class_name, class_id in CLASS_NAMES.items():
        class_root = source / class_name
        for image_path in class_root.rglob("*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            split = split_from_path(image_path)
            label_path = matching_label(image_path)
            if split is None or label_path is None:
                continue
            # Prefix with the movement folder to avoid duplicate frame names.
            movement = image_path.relative_to(class_root).parts[0].replace(" ", "_").lower()
            stem = f"{class_name.lower()}_{movement}_{image_path.stem}"
            target_image = destination / "images" / split / f"{stem}{image_path.suffix.lower()}"
            target_label = destination / "labels" / split / f"{stem}.txt"
            shutil.copy2(image_path, target_image)
            rewrite_label(label_path, target_label, class_id)
            counts[f"{split}:{class_name}"] += 1

    yaml = (
        f"path: {destination.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [17, 3]\n"
        "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n"
        "names:\n  0: deadlift\n  1: lateral_raise\n  2: squat\n"
    )
    (destination / "exercise_pose.yaml").write_text(yaml, encoding="utf-8")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("JIM_DATA29"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("exercise_pose_dataset"))
    args = parser.parse_args()
    result = build(args.source, args.output)
    print("Prepared YOLO pose dataset:")
    for key, count in sorted(result.items()):
        print(f"  {key}: {count}")
