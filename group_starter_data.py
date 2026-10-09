"""Conservatively cluster visual near-duplicates before any train/test split.

Downloaded-file grouping is weaker than subject-disjoint validation. This
additional screen prevents obvious overlapping recordings being split apart;
it cannot establish all upstream YouTube recording identities.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_starter_data import write_records


def hashes(path):
    capture = cv2.VideoCapture(str(path))
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    values = []
    for fraction in (0.15, 0.35, 0.55, 0.75):
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(count * fraction)))
        ok, frame = capture.read()
        if not ok:
            continue
        # 64-bit difference hash. A match requires several frames, not a
        # single gym background or clothing color.
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8))
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        values.append(int.from_bytes(np.packbits(bits).tobytes(), "big"))
    capture.release()
    return values


def run(root):
    rows = json.loads((root / "download_receipt.json").read_text(encoding="utf-8"))
    features = [hashes(row["path"]) for row in rows]
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    merges = []
    for i in range(len(rows)):
        for j in range(i):
            same = rows[i]["group"] == rows[j]["group"] or rows[i]["sha256"] == rows[j]["sha256"]
            close = (len(features[i]) == 4 and len(features[j]) == 4 and
                     sum(min((a ^ b).bit_count() for b in features[j]) <= 6 for a in features[i]) >= 3)
            if same or close:
                parent[find(i)] = find(j)
                if close and not same:
                    merges.append([rows[j]["dataset_path"], rows[i]["dataset_path"]])
    for i, row in enumerate(rows):
        row["group"] = rows[find(i)]["group"]
        row["visual_hashes"] = [hex(x) for x in features[i]]
    write_records(root, rows)
    report = {"downloaded_videos": len(rows), "groups_after_clustering": len({r["group"] for r in rows}),
              "near_duplicate_merges": merges, "subject_disjoint": False,
              "warning": "Kaggle source recordings/person identities not provided; visual clustering is approximate."}
    (root / "grouping_receipt.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
