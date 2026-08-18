"""Pose-detection and squat-form analysis used by the API."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)

# Ultralytics normally stores settings in the current user's roaming profile.
# That location is often read-only on servers, containers, and managed PCs, so
# keep its small settings file beside this application by default. Create the
# root first because Ultralytics checks writability before creating it.
_config_root = Path(__file__).with_name(".ultralytics")
_config_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_config_root))
# Matplotlib may be imported by the model runtime; keep its cache writable too.
os.environ.setdefault("MPLCONFIGDIR", str(_config_root / "matplotlib"))

KEYPOINT_CONFIDENCE = 0.25


def resolve_yolo_device() -> str | int:
    """Pick YOLO inference device: GPU when available, else CPU. Override with YOLO_DEVICE."""
    override = os.getenv("YOLO_DEVICE")
    if override:
        return override
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class InvalidImageError(ValueError):
    """The upload could not be decoded as an image."""


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle ABC in degrees, normalized to the range 0–180."""
    first = math.atan2(float(a[1] - b[1]), float(a[0] - b[0]))
    second = math.atan2(float(c[1] - b[1]), float(c[0] - b[0]))
    angle = abs(math.degrees(first - second))
    return 360.0 - angle if angle > 180.0 else angle


class PoseAnalyzer:
    """Loads YOLO once and converts detections into API-friendly form feedback."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        base_model_path = Path(__file__).with_name("yolov8n-pose.pt")
        exercise_model_path = Path(os.getenv("EXERCISE_POSE_MODEL_PATH", Path(__file__).with_name("exercise_pose_best.pt")))
        # Prefer the fine-tuned exercise pose model when available. Passing an
        # explicit POSE_MODEL_PATH still overrides this default.
        default_path = exercise_model_path if exercise_model_path.is_file() else base_model_path
        self.model_path = Path(model_path or default_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Pose model not found: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(f"The pose runtime import error: {error}. Run: pip install -r requirements.txt") from error
        self.device = resolve_yolo_device()
        self.model = YOLO(str(self.model_path))
        self.is_exercise_pose_model = self.model_path == exercise_model_path
        self.exercise_classifier = self._load_exercise_classifier()

    @staticmethod
    def _load_exercise_classifier() -> Any | None:
        """Load the optional angle-based exercise classifier trained from the CSV dataset."""
        classifier_path = Path(__file__).with_name("exercise_classifier.joblib")
        if not classifier_path.is_file():
            return None
        try:
            import joblib

            return joblib.load(classifier_path)
        except (ImportError, OSError, ValueError):
            return None

    def process_frame(self, image_bytes: bytes, confidence: float = 0.35) -> dict[str, Any]:
        """Analyse an encoded image and return the most prominent person's pose."""
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise InvalidImageError("The uploaded file is not a valid image.")

        result = self.model.predict(frame, conf=confidence, verbose=False, imgsz=640, device=self.device)[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return {
                "detected": False,
                "image": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
                "message": "No person pose was detected. Use a well-lit, full-body image.",
                "people": [],
            }

        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(result.keypoints.xy))
        person_index = int(np.argmax(confidences))
        points = result.keypoints.xy[person_index].cpu().numpy()
        point_confidences = result.keypoints.conf[person_index].cpu().numpy() if result.keypoints.conf is not None else np.ones(len(points))

        landmarks = {
            name: {
                "x": round(float(point[0]), 2),
                "y": round(float(point[1]), 2),
                "confidence": round(float(point_confidences[index]), 3),
            }
            for index, (name, point) in enumerate(zip(COCO_KEYPOINTS, points))
        }
        exercise_label = None
        if self.is_exercise_pose_model and result.boxes is not None:
            class_id = int(result.boxes.cls[person_index].item())
            exercise_label = str(self.model.names[class_id])
        person = self._analyse_person(
            points, point_confidences, float(confidences[person_index]), landmarks, exercise_label
        )
        return {
            "detected": True,
            "image": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "people_detected": int(len(result.keypoints.xy)),
            "people": [person],
        }

    def _analyse_person(
        self, points: np.ndarray, point_confidences: np.ndarray, detection_confidence: float,
        landmarks: dict[str, dict[str, float]], exercise_label: str | None = None,
    ) -> dict[str, Any]:
        posture = self._posture_metrics(points, point_confidences)
        exercise = self._classify_exercise(points, point_confidences)
        
        # Decide the final exercise label
        final_label = None
        angle_label = exercise["label"] if (exercise and exercise.get("reliable")) else None
        angle_conf = exercise["confidence"] if exercise else 0.0

        # If the angle classifier is confident in a class that YOLO cannot predict, trust it.
        if angle_label in ("Push Ups", "Jumping Jacks", "Pull ups", "Russian twists") and angle_conf >= 0.70:
            final_label = angle_label
        # Otherwise, if YOLO predicted a label, trust YOLO.
        elif exercise_label:
            final_label = exercise_label
            exercise = {
                "label": exercise_label,
                "confidence": round(detection_confidence, 3),
                "reliable": detection_confidence >= 0.55,
                "source": "fine_tuned_yolo_pose",
            }
        else:
            final_label = angle_label

        label_lower = final_label.lower() if final_label else "unknown"

        knee_angles: dict[str, float] = {}
        elbow_angles: dict[str, float] = {}
        for side, hip_index, knee_index, ankle_index in (("left", 11, 13, 15), ("right", 12, 14, 16)):
            if min(point_confidences[hip_index], point_confidences[knee_index], point_confidences[ankle_index]) >= KEYPOINT_CONFIDENCE:
                knee_angles[side] = round(calculate_angle(points[hip_index], points[knee_index], points[ankle_index]), 1)

        feedback: list[str] = []
        if "push up" in label_lower or "pushups" in label_lower:
            for side, shoulder_idx, elbow_idx, wrist_idx in (("left", 5, 7, 9), ("right", 6, 8, 10)):
                if min(point_confidences[shoulder_idx], point_confidences[elbow_idx], point_confidences[wrist_idx]) >= KEYPOINT_CONFIDENCE:
                    elbow_angles[side] = round(calculate_angle(points[shoulder_idx], points[elbow_idx], points[wrist_idx]), 1)
            
            if not elbow_angles:
                feedback.append("Keep your shoulders, elbows, and wrists visible for pushup-depth feedback.")
                depth = "unknown"
            else:
                average_elbow = sum(elbow_angles.values()) / len(elbow_angles)
                if average_elbow <= 100:
                    depth = "deep"
                    feedback.append("Excellent pushup depth (elbow bend under 100°).")
                elif average_elbow <= 135:
                    depth = "parallel"
                    feedback.append("Good pushup depth; try to go slightly lower for full range of motion.")
                else:
                    depth = "shallow"
                    feedback.append("Pushup is shallow; try bending your elbows more to lower your chest.")

            hip_angles: dict[str, float] = {}
            for side, shoulder_idx, hip_idx, knee_idx in (("left", 5, 11, 13), ("right", 6, 12, 14)):
                if min(point_confidences[shoulder_idx], point_confidences[hip_idx], point_confidences[knee_idx]) >= KEYPOINT_CONFIDENCE:
                    hip_angles[side] = calculate_angle(points[shoulder_idx], points[hip_idx], points[knee_idx])
            if hip_angles:
                avg_hip = sum(hip_angles.values()) / len(hip_angles)
                if avg_hip < 160:
                    feedback.append("Keep your torso and hips aligned; brace your core to prevent sagging.")
        elif "jumping jacks" in label_lower:
            depth = "unknown"
            feedback.append("Jumping jacks detected. Ensure you raise your arms fully above your head.")
        elif "pull ups" in label_lower:
            depth = "unknown"
            feedback.append("Pull-ups detected. Pull until your chin clears the bar, keeping your core stable.")
        elif "deadlift" in label_lower:
            depth = "unknown"
            feedback.append("Deadlift detected. Focus on driving through your heels and keeping a flat back.")
        elif "lateral_raise" in label_lower or "lateral raise" in label_lower:
            depth = "unknown"
            feedback.append("Lateral raises detected. Lift weight to shoulder height; keep elbows slightly bent.")
        else: # Default is Squat analysis
            if not knee_angles:
                feedback.append("Keep your hips, knees, and ankles visible for squat-depth feedback.")
                depth = "unknown"
            else:
                average_angle = sum(knee_angles.values()) / len(knee_angles)
                if average_angle <= 100:
                    depth = "deep"
                    feedback.append("Good squat depth detected.")
                elif average_angle <= 135:
                    depth = "parallel"
                    feedback.append("You are approaching parallel; lower slightly for a deeper squat if comfortable.")
                else:
                    depth = "shallow"
                    feedback.append("Squat appears shallow; bend your knees and sit back further if appropriate.")
            feedback.extend(self._posture_feedback(posture))

        visible_keypoints = int(np.count_nonzero(point_confidences >= KEYPOINT_CONFIDENCE))
        if visible_keypoints < 11:
            feedback.append("For a more reliable assessment, use a clear, full-body side or front view.")

        return {
            "confidence": round(detection_confidence, 3),
            "landmarks": landmarks,
            "metrics": {
                "knee_angles": knee_angles,
                "elbow_angles": elbow_angles,
                "squat_depth": depth,
                "posture": posture,
                "visible_keypoints": visible_keypoints,
                "exercise": exercise,
            },
            "feedback": feedback,
        }

    @staticmethod
    def _posture_metrics(points: np.ndarray, confidences: np.ndarray) -> dict[str, float | str | None]:
        """Estimate symmetry and torso alignment only when needed landmarks are reliable."""
        metrics: dict[str, float | str | None] = {
            "knee_angle_difference": None,
            "torso_lean_degrees": None,
            "knee_tracking": "unknown",
        }
        left_leg = (11, 13, 15)
        right_leg = (12, 14, 16)
        if min(confidences[index] for index in (*left_leg, *right_leg)) >= KEYPOINT_CONFIDENCE:
            left_angle = calculate_angle(points[left_leg[0]], points[left_leg[1]], points[left_leg[2]])
            right_angle = calculate_angle(points[right_leg[0]], points[right_leg[1]], points[right_leg[2]])
            metrics["knee_angle_difference"] = round(abs(left_angle - right_angle), 1)

            # In a front-facing image, a knee should remain roughly over its ankle.
            # Scale by hip width so this remains meaningful at different distances.
            hip_width = abs(float(points[11][0] - points[12][0]))
            if hip_width > 8:
                offsets = [abs(float(points[knee][0] - points[ankle][0])) / hip_width for knee, ankle in ((13, 15), (14, 16))]
                metrics["knee_tracking"] = "aligned" if max(offsets) < 0.55 else "check_alignment"

        shoulder_indices, hip_indices = (5, 6), (11, 12)
        if min(confidences[index] for index in (*shoulder_indices, *hip_indices)) >= KEYPOINT_CONFIDENCE:
            shoulder_mid = (points[5] + points[6]) / 2
            hip_mid = (points[11] + points[12]) / 2
            vector = shoulder_mid - hip_mid
            # Angle away from vertical; a side view produces the most useful value.
            metrics["torso_lean_degrees"] = round(abs(math.degrees(math.atan2(float(vector[0]), -float(vector[1])))), 1)
        return metrics

    @staticmethod
    def _posture_feedback(posture: dict[str, float | str | None]) -> list[str]:
        feedback: list[str] = []
        asymmetry = posture["knee_angle_difference"]
        if isinstance(asymmetry, float) and asymmetry > 18:
            feedback.append("Your knee bend looks uneven side-to-side; aim for a more balanced descent.")
        if posture["knee_tracking"] == "check_alignment":
            feedback.append("Check that your knees track in the same direction as your toes.")
        lean = posture["torso_lean_degrees"]
        if isinstance(lean, float) and lean > 55:
            feedback.append("Your torso is leaning markedly; keep your chest controlled and brace your core.")
        return feedback

    def _classify_exercise(self, points: np.ndarray, confidences: np.ndarray) -> dict[str, Any] | None:
        """Classify exercise from the four joint angles available in a COCO pose."""
        if self.exercise_classifier is None:
            return None
        samples: list[dict[str, float | str]] = []
        for side, shoulder, elbow, wrist, hip, knee, ankle in (
            ("left", 5, 7, 9, 11, 13, 15),
            ("right", 6, 8, 10, 12, 14, 16),
        ):
            required = (shoulder, elbow, wrist, hip, knee, ankle)
            if min(confidences[index] for index in required) < KEYPOINT_CONFIDENCE:
                continue
            samples.append(
                {
                    "Side": side,
                    "Shoulder_Angle": calculate_angle(points[elbow], points[shoulder], points[hip]),
                    "Elbow_Angle": calculate_angle(points[shoulder], points[elbow], points[wrist]),
                    "Hip_Angle": calculate_angle(points[shoulder], points[hip], points[knee]),
                    "Knee_Angle": calculate_angle(points[hip], points[knee], points[ankle]),
                }
            )
        if not samples:
            return None

        model = self.exercise_classifier["model"]
        probabilities = model.predict_proba(samples)
        averaged = probabilities.mean(axis=0)
        class_index = int(np.argmax(averaged))
        confidence = float(averaged[class_index])
        candidate = str(model.classes_[class_index])
        return {
            "label": candidate if confidence >= 0.55 else "unknown",
            "candidate": candidate,
            "confidence": round(confidence, 3),
            "reliable": confidence >= 0.55,
            "sides_used": len(samples),
        }
