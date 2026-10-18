"""Merge extracted pose manifests while preserving their source groups.

The extractor writes clip paths relative to each output directory. This
utility emits absolute NPZ paths so a merged manifest can live outside those
directories without changing provenance or accidentally merging groups.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


def merge(output: Path, inputs: Sequence[Path], exclude_labels: Sequence[str] = ()) -> int:
    rows: list[tuple[str, str, str]] = []
    seen: set[Path] = set()
    excluded = set(exclude_labels)
    for manifest in inputs:
        manifest = Path(manifest).resolve()
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"clip_path", "label", "group"}.issubset(reader.fieldnames):
                raise ValueError(f"{manifest}: manifest requires clip_path,label,group columns")
            for line, row in enumerate(reader, 2):
                raw = (row.get("clip_path") or "").strip()
                label = (row.get("label") or "").strip()
                group = (row.get("group") or "").strip()
                if not raw or not label or not group:
                    raise ValueError(f"{manifest}:{line}: clip_path, label, and group are required")
                if label in excluded:
                    continue
                clip = (manifest.parent / raw).resolve()
                if clip in seen:
                    raise ValueError(f"duplicate clip path: {clip}")
                if not clip.is_file():
                    raise FileNotFoundError(f"pose clip does not exist: {clip}")
                seen.add(clip)
                rows.append((str(clip), label, group))

    if not rows:
        raise ValueError("no pose clips found in the input manifests")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("clip_path", "label", "group"))
        writer.writerows(rows)
    print(f"merged {len(rows)} clips from {len(inputs)} manifests into {output}")
    return len(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=[],
        help="Exclude an exact label from all input manifests; may be repeated",
    )
    args = parser.parse_args(argv)
    try:
        merge(args.output, args.manifests, args.exclude_label)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
