"""Experimental v3 temporal pose classifier and conservative rep tracking."""

from __future__ import annotations

import math
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
from pose_features_v3 import extract_window_features, landmarks_to_xyz
from repetition_counter_v3 import RepetitionCounterV3, visible_angles
from train_pose_v3 import predict_hierarchy


MODEL_TYPE = "hierarchical_decision_tree"
FEATURE_TYPE = "normalized_mediapipe_temporal_20"
FEATURE_SIZE = 792
WINDOW_SECONDS = 1.9
SAMPLING_FPS = 10
VISIBILITY_THRESHOLD = 0.35


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
    """Image-mode MediaPipe inference backed solely by the v3 model artifact."""

    ACTIONS = tuple(ACTIONS)

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        classification_threshold: float = 0.7,
        model_path: Optional[Path] = None,
        detector: Any = None,
    ) -> None:
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
        self.family_map = dict(FAMILY_MAP)
        self.sequence: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.last_timestamp: Optional[float] = None
        self.counter = RepetitionCounterV3(ACTIONS)
        self.counters = self.counter.counters
        self.stages = self.counter.stages
        self.current_action = "unknown"
        self.current_confidence = 0.0
        self.last_probabilities = {action: 0.0 for action in ACTIONS}
        self.candidate_action = "unknown"
        self.candidate_frames = 0
        self.action_stability_frames = 3
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
        if self.detector is not None and hasattr(self.detector, "close"):
            self.detector.close()
        self.detector = None

    def update_settings(
        self,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        classification_threshold: Optional[float] = None,
    ) -> None:
        changed_detector = min_detection_confidence is not None or min_tracking_confidence is not None
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

    def _clear_tracking(self) -> None:
        self.sequence.clear()
        self.timestamps.clear()
        self.last_timestamp = None
        self.current_action = "unknown"
        self.current_confidence = 0.0
        self.last_probabilities = {action: 0.0 for action in ACTIONS}
        self.candidate_action = "unknown"
        self.candidate_frames = 0
        self.counter.clear_cycle()
        self.stages = self.counter.stages

    def reset_counters(self) -> dict[str, int]:
        self.counter.reset()
        self.counters = self.counter.counters
        self._clear_tracking()
        return self.counters.copy()

    def _payload(self, detected: bool, image: dict[str, int], landmarks: list[dict], angles: dict, rep_event=None, message=None):
        action = self.current_action
        result = {
            "detected": detected,
            "image": image,
            "current_action": action,
            "display_name": action.replace("_", " ").title() if action != "unknown" else "Unknown",
            "family": ACTION_FAMILY.get(action, "unknown"),
            "equipment": "unknown",
            "equipment_detection": "not_trained",
            "experimental": True,
            "model_info": self.model_info,
            "rep_counting": "experimental_heuristic_not_validated",
            "confidence": self.current_confidence,
            "active_stage": self.stages.get(action),
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
        """Classify the last 1.9 seconds of sufficiently sampled, visible poses."""
        if timestamp_seconds is not None and (
            isinstance(timestamp_seconds, bool) or not isinstance(timestamp_seconds, (int, float))
            or not math.isfinite(timestamp_seconds) or timestamp_seconds < 0
        ):
            raise ValueError("timestamp_seconds must be finite and nonnegative")
        timestamp = float(timestamp_seconds) if timestamp_seconds is not None else time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self._clear_tracking()
            return self._payload(False, {}, [], {}, message="Invalid image payload")
        height, width = frame.shape[:2]
        image = {"width": width, "height": height}
        if self.last_timestamp is not None and (timestamp <= self.last_timestamp or timestamp - self.last_timestamp > 0.35):
            self._clear_tracking()
        self.last_timestamp = timestamp
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
        if not detection.pose_landmarks:
            self._clear_tracking()
            return self._payload(False, image, [], {}, message="No visible pose")
        pose = detection.pose_landmarks[0]
        try:
            xyz = landmarks_to_xyz(pose, visibility_threshold=VISIBILITY_THRESHOLD)
            if xyz is None or np.asarray(xyz).shape != (33, 3) or not np.isfinite(xyz).all():
                raise ValueError("Insufficient visible pose")
        except (ValueError, TypeError, AttributeError, IndexError):
            self._clear_tracking()
            return self._payload(False, image, [], {}, message="Insufficient visible pose")
        landmarks = [
            {"id": index, "x": round(float(lm.x) * width, 2),
             "y": round(float(lm.y) * height, 2), "z": round(float(lm.z), 3),
             "visibility": round(float(getattr(lm, "visibility", 0)), 3)}
            for index, lm in enumerate(pose)
        ]
        angles = visible_angles(pose, VISIBILITY_THRESHOLD)
        self.sequence.append(np.asarray(xyz, dtype=np.float64))
        self.timestamps.append(timestamp)
        # Retain the sample immediately before the window boundary so that
        # interpolation covers the entire 1.9-second interval.
        while len(self.timestamps) > 1 and timestamp - self.timestamps[1] >= WINDOW_SECONDS:
            self.timestamps.pop(0)
            self.sequence.pop(0)
        if len(self.sequence) < 20 or self.timestamps[-1] - self.timestamps[0] < WINDOW_SECONDS - 1e-9:
            self.current_action = "unknown"
            self.counter.clear_cycle()
            self.stages = self.counter.stages
            return self._payload(True, image, landmarks, angles, message="Buffering pose window")
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
        if (action == "unknown" or confidence < max(self.classification_threshold, self.artifact["threshold"], 0.7)
                or margin < self.artifact["margin"]):
            self._clear_tracking()
            return self._payload(True, image, landmarks, angles, message="Low-confidence movement")
        if action != self.candidate_action:
            self.candidate_action = action
            self.candidate_frames = 1
            self.current_action = "unknown"
            self.counter.clear_cycle()
            self.stages = self.counter.stages
        else:
            self.candidate_frames += 1
        if self.candidate_frames < self.action_stability_frames:
            return self._payload(True, image, landmarks, angles, message="Stabilizing movement")
        self.current_action = action
        rep_event = self.counter.update(action, angles, timestamp)
        self.stages = self.counter.stages
        return self._payload(True, image, landmarks, angles, rep_event)
