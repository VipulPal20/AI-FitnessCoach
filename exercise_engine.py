"""AI Exercise Recognition & Rep Counter Engine.
Implements MediaPipe Pose Landmarker, Bidirectional Attention-LSTM Deep Learning Model,
Kinematic Joint Angle Extraction, and Repetition Counter Heuristics.
Based on the architecture by Chris Prasanna (Exercise_Recognition_AI).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Flatten,
    Input,
    Permute,
    multiply,
)
from tensorflow.keras.models import Model


def attention_block(inputs, time_steps):
    """Attention layer for deep neural network."""
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation="softmax")(a)
    a_probs = Permute((2, 1), name="attention_vec")(a)
    output_attention_mul = multiply([inputs, a_probs], name="attention_mul")
    return output_attention_mul


def build_attention_lstm_model(
    hidden_units: int = 256,
    sequence_length: int = 30,
    num_input_values: int = 33 * 4,
    num_classes: int = 3,
    weights_path: Optional[Path] = None,
) -> Model:
    """Builds and loads the Attention-LSTM model."""
    inputs = Input(shape=(sequence_length, num_input_values))
    lstm_out = Bidirectional(LSTM(hidden_units, return_sequences=True))(inputs)
    attention_mul = attention_block(lstm_out, sequence_length)
    attention_mul = Flatten()(attention_mul)
    x = Dense(2 * hidden_units, activation="relu")(attention_mul)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=[inputs], outputs=outputs)

    if weights_path and weights_path.is_file():
        try:
            model.load_weights(str(weights_path))
            print(f"Loaded weights from {weights_path}")
        except Exception as err:
            print(f"Warning: Failed to load weights from {weights_path}: {err}")
    return model


class ExerciseAI:
    """Core AI processing engine for exercise recognition and rep tracking."""

    ACTIONS = np.array(["curl", "press", "squat"])

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        classification_threshold: float = 0.5,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.classification_threshold = classification_threshold
        self.sequence_length = 30
        self.sequence: List[np.ndarray] = []

        # Rep Counters & Stages
        self.counters = {"curl": 0, "press": 0, "squat": 0}
        self.stages = {"curl": None, "press": None, "squat": None}
        self.current_action: str = ""
        self.current_confidence: float = 0.0
        self.last_probabilities: Dict[str, float] = {"curl": 0.0, "press": 0.0, "squat": 0.0}

        # Initialize MediaPipe Pose Landmarker Task
        base_dir = Path(__file__).parent
        task_model_path = base_dir / "models" / "pose_landmarker.task"
        if not task_model_path.is_file():
            raise FileNotFoundError(f"Pose landmarker model not found at {task_model_path}")

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Load Deep Learning Weights
        weights_att = base_dir / "models" / "LSTM_Attention.h5"
        weights_lstm = base_dir / "models" / "LSTM.h5"
        chosen_weights = weights_att if weights_att.is_file() else weights_lstm

        self.model = build_attention_lstm_model(weights_path=chosen_weights)
        self.model_loaded = chosen_weights.is_file()

    def update_settings(
        self,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        classification_threshold: Optional[float] = None,
    ) -> None:
        """Update detection and classification threshold settings."""
        if min_detection_confidence is not None:
            self.min_detection_confidence = min_detection_confidence
        if min_tracking_confidence is not None:
            self.min_tracking_confidence = min_tracking_confidence
        if classification_threshold is not None:
            self.classification_threshold = classification_threshold

        base_dir = Path(__file__).parent
        task_model_path = base_dir / "models" / "pose_landmarker.task"
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def reset_counters(self) -> Dict[str, int]:
        """Reset rep counters and stages."""
        self.counters = {"curl": 0, "press": 0, "squat": 0}
        self.stages = {"curl": None, "press": None, "squat": None}
        self.sequence.clear()
        self.current_action = ""
        self.current_confidence = 0.0
        return self.counters.copy()

    @staticmethod
    def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate 2D/3D planar joint angle ABC (degrees, 0-180) around vertex b."""
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360.0 - angle
        return float(angle)

    def extract_keypoints(self, landmarks) -> np.ndarray:
        """Extract 33 * 4 (x, y, z, visibility) normalized features."""
        if landmarks and len(landmarks) >= 33:
            return np.array(
                [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0) or 1.0] for lm in landmarks]
            ).flatten()
        return np.zeros(33 * 4)

    def count_reps_and_angles(self, landmarks) -> Tuple[Dict[str, float], Optional[str]]:
        """Extract joint angles and compute rep counter updates."""
        angles: Dict[str, float] = {}
        rep_event: Optional[str] = None

        try:
            # Helper to get (x,y)
            def pt(idx: int) -> Tuple[float, float]:
                return (landmarks[idx].x, landmarks[idx].y)

            # 1. Arm Angles (Elbows)
            # 11: left_shoulder, 13: left_elbow, 15: left_wrist
            # 12: right_shoulder, 14: right_elbow, 16: right_wrist
            l_shoulder, l_elbow, l_wrist = pt(11), pt(13), pt(15)
            r_shoulder, r_elbow, r_wrist = pt(12), pt(14), pt(16)

            l_elbow_angle = round(self.calculate_angle(l_shoulder, l_elbow, l_wrist), 1)
            r_elbow_angle = round(self.calculate_angle(r_shoulder, r_elbow, r_wrist), 1)
            angles["left_elbow"] = l_elbow_angle
            angles["right_elbow"] = r_elbow_angle

            # 2. Leg Angles (Hips & Knees)
            # 23: left_hip, 25: left_knee, 27: left_ankle
            # 24: right_hip, 26: right_knee, 28: right_ankle
            l_hip, l_knee, l_ankle = pt(23), pt(25), pt(27)
            r_hip, r_knee, r_ankle = pt(24), pt(26), pt(28)

            l_knee_angle = round(self.calculate_angle(l_hip, l_knee, l_ankle), 1)
            r_knee_angle = round(self.calculate_angle(r_hip, r_knee, r_ankle), 1)
            l_hip_angle = round(self.calculate_angle(l_shoulder, l_hip, l_knee), 1)
            r_hip_angle = round(self.calculate_angle(r_shoulder, r_hip, r_knee), 1)

            angles["left_knee"] = l_knee_angle
            angles["right_knee"] = r_knee_angle
            angles["left_hip"] = l_hip_angle
            angles["right_hip"] = r_hip_angle

            # Active Exercise Rep Counter Heuristics (matches Chris Prasanna's logic)
            if self.current_action == "curl":
                active_angle = min(l_elbow_angle, r_elbow_angle)
                if active_angle < 35:
                    self.stages["curl"] = "up"
                if active_angle > 140 and self.stages["curl"] == "up":
                    self.stages["curl"] = "down"
                    self.counters["curl"] += 1
                    rep_event = "curl"
                self.stages["press"] = None
                self.stages["squat"] = None

            elif self.current_action == "press":
                active_elbow = (l_elbow_angle + r_elbow_angle) / 2.0
                l_s2e = math.dist(l_shoulder, l_elbow)
                l_s2w = math.dist(l_shoulder, l_wrist)
                r_s2e = math.dist(r_shoulder, r_elbow)
                r_s2w = math.dist(r_shoulder, r_wrist)

                if (active_elbow > 130) and (l_s2e < l_s2w or r_s2e < r_s2w):
                    self.stages["press"] = "up"
                if (active_elbow < 60) and (l_s2e > l_s2w or r_s2e > r_s2w) and (self.stages["press"] == "up"):
                    self.stages["press"] = "down"
                    self.counters["press"] += 1
                    rep_event = "press"
                self.stages["curl"] = None
                self.stages["squat"] = None

            elif self.current_action == "squat":
                thr = 155
                if (
                    l_knee_angle < thr
                    and r_knee_angle < thr
                    and l_hip_angle < thr
                    and r_hip_angle < thr
                ):
                    self.stages["squat"] = "down"
                if (
                    l_knee_angle > thr
                    and r_knee_angle > thr
                    and l_hip_angle > thr
                    and r_hip_angle > thr
                    and self.stages["squat"] == "down"
                ):
                    self.stages["squat"] = "up"
                    self.counters["squat"] += 1
                    rep_event = "squat"
                self.stages["curl"] = None
                self.stages["press"] = None

        except Exception as e:
            pass

        return angles, rep_event

    def process_frame(self, image_bytes: bytes) -> Dict[str, Any]:
        """Process an uploaded or streaming image frame, recognizing exercise & counting reps."""
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"detected": False, "message": "Invalid image payload"}

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            return {
                "detected": False,
                "image": {"width": w, "height": h},
                "current_action": self.current_action,
                "confidence": self.current_confidence,
                "probabilities": self.last_probabilities,
                "counters": self.counters,
                "stages": self.stages,
                "angles": {},
                "landmarks": [],
                "rep_event": None,
            }

        person_landmarks = detection_result.pose_landmarks[0]

        # Format landmarks for client canvas overlay
        landmarks_list = [
            {
                "id": idx,
                "x": round(lm.x * w, 2),
                "y": round(lm.y * h, 2),
                "z": round(lm.z, 3),
                "visibility": round(getattr(lm, "visibility", 1.0) or 1.0, 3),
            }
            for idx, lm in enumerate(person_landmarks)
        ]

        # Extract features for temporal sequence
        kpts = self.extract_keypoints(person_landmarks)
        self.sequence.append(kpts.astype("float32", casting="same_kind"))
        self.sequence = self.sequence[-self.sequence_length:]

        # Neural Network Classification (LSTM Attention)
        probabilities = self.last_probabilities
        if len(self.sequence) == self.sequence_length and self.model_loaded:
            try:
                input_data = np.expand_dims(self.sequence, axis=0)
                pred = self.model.predict(input_data, verbose=0)[0]
                probabilities = {
                    self.ACTIONS[i]: round(float(pred[i]), 3) for i in range(len(self.ACTIONS))
                }
                self.last_probabilities = probabilities

                max_idx = int(np.argmax(pred))
                conf = float(pred[max_idx])
                action = self.ACTIONS[max_idx]

                if conf >= self.classification_threshold:
                    self.current_action = action
                    self.current_confidence = round(conf, 3)
                else:
                    self.current_action = ""
                    self.current_confidence = round(conf, 3)
            except Exception as e:
                print("Prediction error:", e)

        # Rep counting and joint angle extraction
        angles, rep_event = self.count_reps_and_angles(person_landmarks)

        # Active stage
        active_stage = self.stages.get(self.current_action)

        return {
            "detected": True,
            "image": {"width": w, "height": h},
            "current_action": self.current_action,
            "confidence": self.current_confidence,
            "active_stage": active_stage,
            "probabilities": probabilities,
            "counters": self.counters,
            "stages": self.stages,
            "angles": angles,
            "landmarks": landmarks_list,
            "rep_event": rep_event,
            "buffer_progress": f"{len(self.sequence)}/{self.sequence_length}",
        }
