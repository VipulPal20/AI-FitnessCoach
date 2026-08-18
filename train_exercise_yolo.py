"""Fine-tune YOLOv8 pose to detect exercise class and body keypoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("exercise_pose_dataset") / "exercise_pose.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the dataset to use (0-1].")
    args = parser.parse_args()

    config_root = Path(__file__).with_name(".ultralytics")
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
    os.environ.setdefault("MPLCONFIGDIR", str(config_root / "matplotlib"))
    from ultralytics import YOLO

    from vision import resolve_yolo_device

    device = resolve_yolo_device()
    print(f"Training on device: {device}")
    model = YOLO(str(Path(__file__).with_name("yolov8n-pose.pt")))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        patience=8,
        project=str(Path(__file__).with_name("runs")),
        name="exercise_pose",
        exist_ok=True,
        pretrained=True,
        fraction=args.fraction,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    output = Path(__file__).with_name("exercise_pose_best.pt")
    output.write_bytes(best.read_bytes())
    print(f"Saved trained exercise pose model to {output}")


if __name__ == "__main__":
    main()
