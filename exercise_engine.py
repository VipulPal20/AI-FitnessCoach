"""Experimental v3 temporal pose classifier and conservative rep tracking.

Classification needs a 1.9-second window of motion. Rep counting runs on joint
angles every frame, either for the exercise the classifier has settled on or
for a target exercise the user picked.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from movement_schema import ACTIONS, ACTION_FAMILY, FAMILY_MAP
from pose_features_v3 import MAX_GAP_SECONDS, extract_window_features, landmarks_to_xyz
from repetition_counter_v3 import RULES, RepetitionCounterV3, visible_angles
from train_pose_v3 import predict_hierarchy


MODEL_TYPE = "hierarchical_decision_tree"
FEATURE_TYPE = "normalized_mediapipe_temporal_20"
FEATURE_SIZE = 792
WINDOW_SECONDS = 1.9
SAMPLING_FPS = 10
VISIBILITY_THRESHOLD = 0.35
# Gaps up to this long only rebuild the classifier window; rep progress is kept.
TRACKING_RESET_SECONDS = 2.0
REP_ACTIONS = tuple(RULES)


def _model_classes(model, name: str) -> set[str]:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    classes = getattr(estimator, "classes_", None)
    if classes is None or not hasattr(model, "predict_proba"):
        raise ValueError(f"v3 {name} must be a fitted probabilistic classifier")
    labels = [str(label) for label in classes]
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError(f"v3 {name} has invalid classes")
    size = getattr(model, "n_features_in_", getattr(estimator, "n_features_in_", None))
    if size != FEATURE_SIZE:
        raise ValueError(f"v3 {name} expects {size} features, not {FEATURE_SIZE}")
    return set(labels)


def _validate_artifact(artifact: Any) -> None:
    """Refuse legacy/mismatched artifacts before a frame can be classified."""
    if not isinstance(artifact, dict):
        raise ValueError("v3 model artifact must be a dictionary")
    expected = {
        "schema_version": 3,
        "model_type": MODEL_TYPE,
        "feature_type": FEATURE_TYPE,
        "feature_size": FEATURE_SIZE,
        "window_seconds": WINDOW_SECONDS,
        "sampling_fps": SAMPLING_FPS,
        "equipment_detection": "not_trained",
        "production_ready": False,
    }
    for field, value in expected.items():
        if type(artifact.get(field)) is not type(value) or artifact[field] != value:
            raise ValueError(f"v3 model artifact requires {field}={value!r}")
    labels = artifact.get("labels")
    if not isinstance(labels, (list, tuple)) or tuple(labels) != tuple(ACTIONS) or len(labels) != 11:
        raise ValueError("v3 model labels must match the 11 canonical actions")
    family_map = artifact.get("family_map")
    if not isinstance(family_map, dict) or family_map != FAMILY_MAP:
        raise ValueError("v3 model family_map must match the canonical family map")
    family_classes = _model_classes(artifact.get("family_model"), "family_model")
    families = set(family_map)
    if family_classes != families:
        raise ValueError("v3 family_model classes do not match family_map")
    exercise_models = artifact.get("exercise_models")
    expected_children = {family for family, labels in family_map.items() if len(labels) > 1}
    if not isinstance(exercise_models, dict) or set(exercise_models) != expected_children:
        raise ValueError("v3 exercise_models must contain exactly the multi-action families")
    for family, model in exercise_models.items():
        if family not in families:
            raise ValueError(f"v3 exercise model has unexpected family {family!r}")
        classes = _model_classes(model, f"exercise_models[{family}]")
        expected_labels = set(family_map[family])
        if classes != expected_labels:
            raise ValueError(f"v3 exercise model {family!r} has mismatched classes")
    for field in ("threshold", "margin"):
        value = artifact.get(field)
        expected_value = 0.7 if field == "threshold" else 0.15
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != expected_value:
            raise ValueError(f"v3 model artifact requires {field}={expected_value}")


class ExerciseAI:
    """Image-mode MediaPipe inference backed solely by the v3 model artifact.

    Public methods are serialized with a lock: the MediaPipe detector and the
    temporal buffers are not safe to share between concurrent requests.
    """

    ACTIONS = tuple(ACTIONS)
    REP_ACTIONS = REP_ACTIONS

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        classification_threshold: float = 0.7,
        model_path: Optional[Path] = None,
        detector: Any = None,
    ) -> None:
        self.lock = threading.RLock()
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.classification_threshold = classification_threshold
        self.model_path = Path(model_path) if model_path is not None else Path(__file__).parent / "models" / "exercise_starter_v3.joblib"
        if not self.model_path.is_file():
            raise FileNotFoundError(f"v3 movement model not found at {self.model_path}")
        self.artifact = joblib.load(self.model_path)
        _validate_artifact(self.artifact)
        self.model_loaded = True
        self.model_info = {
            "schema_version": 3,
            "model_type": MODEL_TYPE,
            "feature_type": FEATURE_TYPE,
            "feature_size": FEATURE_SIZE,
            "window_seconds": WINDOW_SECONDS,
            "sampling_fps": SAMPLING_FPS,
            "threshold": self.artifact["threshold"],
            "margin": self.artifact["margin"],
            "production_ready": False,
            "experimental": True,
            "rep_counting": "experimental_heuristic_not_validated",
        }
        self.model_info["training_metadata"] = self.artifact.get("training_metadata", {})
        self.family_map = dict(FAMILY_MAP)
        self.sequence: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.last_timestamp: Optional[float] = None
        self.last_valid_timestamp: Optional[float] = None
        self.counter = RepetitionCounterV3(ACTIONS)
        self.counters = self.counter.counters
        self.stages = self.counter.stages
        self.current_action = "unknown"
        self.current_confidence = 0.0
        self.last_probabilities = {action: 0.0 for action in ACTIONS}
        self.candidate_action = "unknown"
        self.candidate_frames = 0
        # Frames of agreement needed to adopt an action from "unknown", and the
        # larger number needed to switch away from an established action, so a
        # brief squat/lunge flicker does not wipe out a rep in progress.
        self.action_stability_frames = 3
        self.action_switch_frames = 6
        self.target_action: Optional[str] = None
        self._owns_detector = detector is None
        self.detector = detector if detector is not None else self._create_detector()

    def _create_detector(self):
        task_path = Path(__file__).parent / "models" / "pose_landmarker.task"
        if not task_path.is_file():
            raise FileNotFoundError(f"Pose landmarker model not found at {task_path}")
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        with self.lock:
            if self.detector is not None and hasattr(self.detector, "close"):
                self.detector.close()
            self.detector = None

    def update_settings(
        self,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        classification_threshold: Optional[float] = None,
    ) -> None:
        with self.lock:
            changed_detector = (
                (min_detection_confidence is not None and min_detection_confidence != self.min_detection_confidence)
                or (min_tracking_confidence is not None and min_tracking_confidence != self.min_tracking_confidence)
            )
            if min_detection_confidence is not None:
                self.min_detection_confidence = min_detection_confidence
            if min_tracking_confidence is not None:
                self.min_tracking_confidence = min_tracking_confidence
            if classification_threshold is not None:
                self.classification_threshold = classification_threshold
            if changed_detector and self._owns_detector:
                replacement = self._create_detector()
                old_detector = self.detector
                self.detector = replacement
                if old_detector is not None:
                    old_detector.close()
                self._clear_tracking()

    def set_target(self, action: Optional[str]) -> Optional[str]:
        """Count reps for a chosen exercise instead of the classifier's guess."""
        if action is not None and action not in REP_ACTIONS:
            raise ValueError(f"target must be one of {', '.join(REP_ACTIONS)} or null")
        with self.lock:
            if action != self.target_action:
                self.target_action = action
                self.counter.clear_cycle()
                self.stages = self.counter.stages
            return self.target_action

    def _clear_window(self) -> None:
        """Drop the classifier window but keep the current action and rep cycle."""
        self.sequence.clear()
        self.timestamps.clear()
        self.current_confidence = 0.0
        self.last_probabilities = {action: 0.0 for action in ACTIONS}
        self.candidate_action = "unknown"
        self.candidate_frames = 0

    def _clear_tracking(self) -> None:
        self._clear_window()
        self.last_timestamp = None
        self.last_valid_timestamp = None
        self.current_action = "unknown"
        self.counter.clear_cycle()
        self.stages = self.counter.stages

    def reset_counters(self) -> dict[str, int]:
        with self.lock:
            self.counter.reset()
            self.counters = self.counter.counters
            self._clear_tracking()
            return self.counters.copy()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "current_action": self.current_action,
                "confidence": self.current_confidence,
                "counters": self.counters.copy(),
                "stages": self.stages.copy(),
                "probabilities": self.last_probabilities.copy(),
                "target_action": self.target_action,
            }

    @property
    def rep_action(self) -> str:
        return self.target_action or self.current_action

    def _count(self, angles: dict, timestamp: float) -> Optional[str]:
        action = self.rep_action
        if action not in REP_ACTIONS:
            return None
        event = self.counter.update(action, angles, timestamp)
        self.counters = self.counter.counters
        self.stages = self.counter.stages
        return event

    def _payload(self, detected: bool, image: dict[str, int], landmarks: list[dict], angles: dict, rep_event=None, message=None):
        action = self.current_action
        rep_action = self.rep_action
        result = {
            "detected": detected,
            "image": image,
            "current_action": action,
            "display_name": action.replace("_", " ").title() if action != "unknown" else "Unknown",
            "family": ACTION_FAMILY.get(action, "unknown"),
            "target_action": self.target_action,
            "rep_action": rep_action,
            "equipment": "unknown",
            "equipment_detection": "not_trained",
            "experimental": True,
            "model_info": self.model_info,
            "rep_counting": "experimental_heuristic_not_validated",
            "confidence": self.current_confidence,
            "active_stage": self.stages.get(rep_action),
            "probabilities": self.last_probabilities.copy(),
            "counters": self.counters.copy(),
            "stages": self.stages.copy(),
            "angles": angles,
            "landmarks": landmarks,
            "rep_event": rep_event,
            "buffer_progress": f"{min(len(self.sequence), 20)}/20",
        }
        if message is not None:
            result["message"] = message
        return result

    def process_frame(self, image_bytes: bytes, timestamp_seconds: Optional[float] = None) -> dict[str, Any]:
        with self.lock:
            return self._process_frame(image_bytes, timestamp_seconds)

    def _process_frame(self, image_bytes: bytes, timestamp_seconds: Optional[float]) -> dict[str, Any]:
        """Classify the last 1.9 seconds of visible poses and update rep counts."""
        if timestamp_seconds is not None and (
            isinstance(timestamp_seconds, bool) or not isinstance(timestamp_seconds, (int, float))
            or not math.isfinite(timestamp_seconds) or timestamp_seconds < 0
        ):
            raise ValueError("timestamp_seconds must be finite and nonnegative")
        timestamp = float(timestamp_seconds) if timestamp_seconds is not None else time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return self._payload(False, {}, [], {}, message="Invalid image payload")
        height, width = frame.shape[:2]
        image = {"width": width, "height": height}
        if self.last_timestamp is not None:
            step = timestamp - self.last_timestamp
            if step <= 0 or step > TRACKING_RESET_SECONDS:
                # A new clip, a rewind, or a long pause: start fresh.
                self._clear_tracking()
            elif step > MAX_GAP_SECONDS:
                # Too sparse for the classifier window, but a rep can survive it.
                self._clear_window()
        self.last_timestamp = timestamp
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
        if not detection.pose_landmarks:
            return self._pose_lost(image, timestamp, "No visible pose")
        pose = detection.pose_landmarks[0]
        landmarks = [
            {"id": index, "x": round(float(lm.x) * width, 2),
             "y": round(float(lm.y) * height, 2), "z": round(float(lm.z), 3),
             "visibility": round(float(getattr(lm, "visibility", 0)), 3)}
            for index, lm in enumerate(pose)
        ]
        angles = visible_angles(pose, VISIBILITY_THRESHOLD)
        try:
            xyz = landmarks_to_xyz(pose, visibility_threshold=VISIBILITY_THRESHOLD)
            if xyz is None or np.asarray(xyz).shape != (33, 3) or not np.isfinite(xyz).all():
                raise ValueError("Insufficient visible pose")
        except (ValueError, TypeError, AttributeError, IndexError):
            # The skeleton is still worth drawing even if a full limb side is hidden.
            payload = self._pose_lost(image, timestamp, "Partial pose: keep one full side of the body in view")
            payload.update(detected=True, landmarks=landmarks, angles=angles)
            return payload
        self.sequence.append(np.asarray(xyz, dtype=np.float64))
        self.timestamps.append(timestamp)
        self.last_valid_timestamp = timestamp
        # Retain the sample immediately before the window boundary so that
        # interpolation covers the entire 1.9-second interval.
        while len(self.timestamps) > 1 and timestamp - self.timestamps[1] >= WINDOW_SECONDS:
            self.timestamps.pop(0)
            self.sequence.pop(0)
        if len(self.sequence) < 20 or self.timestamps[-1] - self.timestamps[0] < WINDOW_SECONDS - 1e-9:
            rep_event = self._count(angles, timestamp)
            return self._payload(True, image, landmarks, angles, rep_event, message="Buffering pose window")
        if np.any(np.diff(np.asarray(self.timestamps, dtype=np.float64)) > MAX_GAP_SECONDS + 1e-9):
            self._clear_window()
            rep_event = self._count(angles, timestamp)
            return self._payload(True, image, landmarks, angles, rep_event, message="Pose gap; rebuilding window")
        try:
            features = extract_window_features(np.asarray(self.sequence), np.asarray(self.timestamps))
            features = np.asarray(features)
            if features.shape != (FEATURE_SIZE,) or not np.isfinite(features).all():
                raise ValueError("v3 feature extraction returned an invalid descriptor")
            probabilities = predict_hierarchy(self.artifact, features)
            if not isinstance(probabilities, dict) or set(probabilities) != set(ACTIONS):
                raise ValueError("v3 classifier returned unexpected action labels")
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > 1
                   for value in probabilities.values()):
                raise ValueError("v3 classifier returned invalid probabilities")
        except (ValueError, TypeError, KeyError) as exc:
            self._clear_tracking()
            raise RuntimeError(f"v3 movement inference failed: {exc}") from exc
        self.last_probabilities = {label: float(probabilities[label]) for label in ACTIONS}
        ranked = sorted(self.last_probabilities.items(), key=lambda item: item[1], reverse=True)
        action, confidence = ranked[0]
        margin = confidence - ranked[1][1]
        self.current_confidence = round(confidence, 3)
        message = None
        confident = (action != "unknown"
                     and confidence >= max(self.classification_threshold, self.artifact["threshold"])
                     and margin >= self.artifact["margin"])
        if not confident:
            # Hold the established action instead of discarding rep progress.
            message = "Low-confidence movement"
        else:
            if action == self.candidate_action:
                self.candidate_frames += 1
            else:
                self.candidate_action = action
                self.candidate_frames = 1
            needed = self.action_stability_frames if self.current_action == "unknown" else self.action_switch_frames
            if action != self.current_action:
                if self.candidate_frames >= needed:
                    self.current_action = action
                else:
                    message = "Stabilizing movement"
        rep_event = self._count(angles, timestamp)
        return self._payload(True, image, landmarks, angles, rep_event, message=message)

    def _pose_lost(self, image: dict[str, int], timestamp: float, message: str) -> dict[str, Any]:
        if self.last_valid_timestamp is None or timestamp - self.last_valid_timestamp > TRACKING_RESET_SECONDS:
            self._clear_tracking()
        elif timestamp - self.last_valid_timestamp > MAX_GAP_SECONDS:
            self._clear_window()
        return self._payload(False, image, [], {}, message=message)
