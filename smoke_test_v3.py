"""Send sampled frames from a local video through the live v3 endpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import uuid
from pathlib import Path
from typing import Sequence
from urllib import error, request

import cv2


def post_frame(url: str, image: bytes, timestamp: float) -> dict:
    boundary = f"----FitnessCoach{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"timestamp_seconds\"\r\n\r\n{timestamp:.3f}\r\n".encode(),
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"frame.jpg\"\r\n"
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode() + image + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    req = request.Request(
        url,
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"frame request failed with HTTP {exc.code}: {detail}") from exc


def run(video_path: Path, url: str, sample_fps: float) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or sample_fps
    stride = max(1, round(source_fps / sample_fps))
    actions: list[str] = []
    messages: Counter[str] = Counter()
    detected_frames = 0
    max_buffer = 0
    last: dict = {}
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                encoded, buffer = cv2.imencode(".jpg", frame)
                if not encoded:
                    raise RuntimeError(f"could not encode frame {frame_index}")
                last = post_frame(url, buffer.tobytes(), frame_index / source_fps)
                data = last.get("data", {})
                actions.append(str(data.get("current_action", "missing")))
                messages[str(data.get("message", "classified"))] += 1
                detected_frames += int(bool(data.get("detected")))
                try:
                    max_buffer = max(max_buffer, int(str(data.get("buffer_progress", "0/20")).split("/", 1)[0]))
                except (ValueError, IndexError):
                    pass
            frame_index += 1
    finally:
        capture.release()
    data = last.get("data", {})
    return {
        "video": str(video_path),
        "sampled_frames": len(actions),
        "detected_frames": detected_frames,
        "unique_actions": sorted(set(actions)),
        "messages": dict(messages),
        "max_buffer_frames": max_buffer,
        "last_action": data.get("current_action"),
        "last_confidence": data.get("confidence"),
        "last_message": data.get("message"),
        "last_buffer_progress": data.get("buffer_progress"),
        "last_counters": data.get("counters"),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8001/api/v1/process_frame")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.video, args.url, args.fps), indent=2))


if __name__ == "__main__":
    main()
