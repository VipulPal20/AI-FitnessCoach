"""Download a bounded research subset, never executable models or source code.

python prepare_starter_data.py TEMP_DIR --per-class 16
Downloads individual Kaggle videos and selected UCF101 ZIP members using HTTP
byte ranges (NOT the whole 7 GB ZIP). Records licenses, source groups and hashes.
The dataset terms do not grant commercial clearance for underlying web media.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path

import requests

UCF_URL = "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip"
KAGGLE_REF = "hasyimabdillah/workoutfitness-video"
LABELS = {
    "bench press": "bench_press", "shoulder press": "shoulder_press",
    "push-up": "push_up", "squat": "squat", "deadlift": "deadlift",
    "barbell biceps curl": "bicep_curl", "pull Up": "pull_up", "t bar row": "row",
}


class RemoteZip(io.RawIOBase):
    """Seekable HTTP ranges; reject servers that silently return the full ZIP."""
    def __init__(self, url):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "FitnessCoachResearch/1.0"
        self.url = url
        r = self.session.get(url, headers={"Range": "bytes=-65536"}, timeout=60)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError("Server must support bounded byte ranges")
        self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.position = 0
        self.cache = [(self.size - len(r.content), r.content)]
        self.downloaded = len(r.content)

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b""
        if count > 16 * 1024 * 1024:
            raise ValueError("Refusing range above 16 MB")
        start = self.position
        end = start + count
        for cached_start, data in self.cache:
            if cached_start <= start and end <= cached_start + len(data):
                self.position = end
                return data[start-cached_start:end-cached_start]
        r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end-1}"}, timeout=60)
        r.raise_for_status()
        if r.status_code != 206 or len(r.content) != count:
            raise RuntimeError("Unexpected HTTP range response")
        self.downloaded += count
        if len(self.cache) < 8:
            self.cache.append((start, r.content))
        self.position = end
        return r.content


def kaggle_files(ref):
    files, token = [], None
    session = requests.Session()
    for _ in range(30):
        params = {"pageSize": 200}
        if token:
            params["pageToken"] = token
        r = session.get("https://www.kaggle.com/api/v1/datasets/list/" + ref, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        files.extend(payload.get("datasetFiles", []))
        token = payload.get("nextPageToken")
        if not token:
            return files
    raise RuntimeError("Exceeded listing pagination limit")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unwrap_video(path):
    """Kaggle sometimes returns ZIP content with an .mp4 filename."""
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        candidates = [i for i in archive.infolist() if not i.is_dir() and Path(i.filename).name == path.name]
        if len(candidates) != 1 or candidates[0].file_size > 80 * 1024 * 1024:
            raise ValueError("Expected one bounded video in Kaggle file archive")
        content = archive.read(candidates[0])
    # Never extract archive paths. Replace only the explicitly downloaded file.
    path.write_bytes(content)
    return True


def repair_archives(root):
    rows = json.loads((root / "download_receipt.json").read_text(encoding="utf-8"))
    count = 0
    for row in rows:
        path = Path(row["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Refusing to modify files outside the owned download root")
        if unwrap_video(path):
            count += 1
            row["bytes"] = path.stat().st_size
            row["sha256"] = digest(path)
            if row["dataset_path"].startswith(tuple(folder + "/" for folder in LABELS)):
                row["group"] = "hasyim_recording_" + row["sha256"][:20]
    write_records(root, rows)
    print("UNWRAPPED", count, "video archives")


def write_records(root, records):
    with (root / "videos.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "group", "source_url", "license"])
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    (root / "download_receipt.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def acquire(root, per_class):
    import kagglehub
    root.mkdir(parents=True, exist_ok=True)
    records = []
    receipt = root / "download_receipt.json"
    if receipt.exists():
        records = json.loads(receipt.read_text(encoding="utf-8"))
    downloaded = {r["dataset_path"] for r in records}
    files = kaggle_files(KAGGLE_REF)
    rng = random.Random(42)
    for folder, label in LABELS.items():
        eligible = [f for f in files if f["name"].startswith(folder + "/")
                    and f["name"].lower().endswith(".mp4") and f["totalBytes"] <= 16 * 1024 * 1024]
        rng.shuffle(eligible)
        for item in eligible[:per_class]:
            name = item["name"]
            if name in downloaded:
                continue
            try:
                path = Path(kagglehub.dataset_download(KAGGLE_REF + "/versions/5", path=name,
                                                      output_dir=str(root / "kaggle")))
                unwrap_video(path)
                sha = digest(path)
                records.append({"path": str(path.resolve()), "label": label,
                                "group": "hasyim_recording_" + sha[:20],
                                "dataset_path": name,
                                "source_url": "https://www.kaggle.com/datasets/" + KAGGLE_REF,
                                "license": "CC BY-NC-SA 4.0; upstream YouTube rights unverified",
                                "bytes": path.stat().st_size, "sha256": sha})
                write_records(root, records)
                print("DOWNLOADED", label, name, flush=True)
            except Exception as exc:
                print("FAILED", label, name, str(exc), flush=True)
    remote = RemoteZip(UCF_URL)
    with zipfile.ZipFile(remote) as archive:
        # gXX identifies original UCF recording groups, not individual clips.
        for source_label, target in [("Lunges", "lunge"), ("JumpingJack", "jumping_jack"),
                                     ("WalkingWithDog", "unknown"), ("ApplyEyeMakeup", "unknown")]:
            groups = {}
            for info in archive.infolist():
                match = re.search(r"v_" + source_label + r"_g(\d+)_c\d+\.avi$", info.filename)
                if match and info.file_size < 8 * 1024 * 1024:
                    groups.setdefault(match[1], []).append(info)
            group_ids = sorted(groups)
            rng.shuffle(group_ids)
            for group in group_ids[:per_class]:
                info = sorted(groups[group], key=lambda v: v.filename)[0]
                if info.filename in downloaded:
                    continue
                path = root / "ucf" / Path(info.filename).name
                path.parent.mkdir(parents=True, exist_ok=True)
                content = archive.read(info)
                path.write_bytes(content)
                records.append({"path": str(path.resolve()), "label": target,
                                "group": f"ucf_{source_label}_g{group}",
                                "dataset_path": info.filename, "source_url": UCF_URL,
                                "license": "UCF101 academic action-recognition benchmark; underlying YouTube rights unverified",
                                "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
                write_records(root, records)
                print("DOWNLOADED", target, info.filename, flush=True)
    print("DOWNLOADS", len(records), "UCF_RANGE_BYTES", remote.downloaded, flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--repair-archives", action="store_true")
    args = parser.parse_args()
    if not 3 <= args.per_class <= 50:
        parser.error("per-class must be between 3 and 50")
    if args.repair_archives:
        repair_archives(args.root.resolve())
    else:
        acquire(args.root.resolve(), args.per_class)
