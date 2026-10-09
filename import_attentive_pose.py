"""Import published pose CSV without downloading videos.

Usage: python import_attentive_pose.py source.csv output_directory
Clip IDs are NOT subject/source-video IDs. This importer is for exploratory
training only; its split cannot establish unseen-person/video generalization.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def convert(source, output):
    clips = defaultdict(list)
    labels = {"BenchPress": "bench_press", "Deadlift": "deadlift", "Squat": "squat"}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        coordinates = [key for key in reader.fieldnames if key.endswith(("_X", "_Y", "_Z"))]
        if len(coordinates) != 99:
            raise ValueError("Expected 99 published MediaPipe coordinate columns")
        for row in reader:
            label = labels[row["Jenis_Latihan"]]
            clip_id = row["Nama_Dataset"]
            if not clip_id.replace("_", "").isalnum():
                raise ValueError("Invalid clip identifier")
            clips[(clip_id, label)].append((int(row["Frame_Ke"]), [float(row[key]) for key in coordinates]))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_path", "label", "group"])
        for (clip_id, label), frames in sorted(clips.items()):
            frames.sort(key=lambda value: value[0])
            if len(frames) != 64 or len({f[0] for f in frames}) != 64:
                raise ValueError(f"Incomplete clip: {clip_id}")
            array = np.asarray([f[1] for f in frames], dtype=np.float32).reshape(64, 33, 3)
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite clip: {clip_id}")
            name = clip_id + ".npz"
            np.savez_compressed(output / name, landmarks=array)
            writer.writerow([name, label, "clip_only_" + clip_id])
    receipt = {"clips": len(clips), "grouping": "clip-only; subjects and original recordings unavailable",
               "production_ready": False, "videos_downloaded": 0,
               "source": "https://www.kaggle.com/datasets/bangaji/attentiveskel3d-weighttraining-dataset-per-frame",
               "license_note": "Kaggle lists MIT; secondary Hasyim data has non-commercial restrictions. Research candidate only."}
    (output / "provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)
