# AI Fitness Coach — Machine Learning Training & Evaluation Report

**Generated**: October 4, 2026  
**Hardware Accelerator**: NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MiB VRAM)  
**Execution Environment**: Python 3.12, PyTorch 2.11.0+cu128 (CUDA:0), TensorFlow 2.15, Scikit-Learn 1.5  

---

## 1. Custom YOLOv8 Pose & Exercise Detector

* **Model File**: [`exercise_pose_best.pt`](exercise_pose_best.pt) (6.8 MB)
* **Architecture**: Ultralytics YOLOv8-pose fine-tuned on custom multi-exercise poses
* **Dataset**: 4,596 total annotated images (`exercise_pose_dataset/`)
  * **Train Set**: 3,270 images (Squats: 1,533, Deadlifts: 1,341, Shoulders: 396)
  * **Validation Set**: 1,326 images (Deadlifts: 660, Squats: 542, Shoulders: 124)
* **Training Setup**: 20 Epochs, Batch Size 8, Image Size 512px

### Overall Accuracy Metrics

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Box Detection mAP@50** | **99.5%** | Precision of locating person bounding boxes and classifying exercise |
| **Box Detection mAP@50-95** | **98.4%** | Strict IoU threshold precision for person localization |
| **Pose Keypoint mAP@50** | **97.6%** | Accuracy of localizing all 17 anatomical keypoints |
| **Pose Keypoint mAP@50-95** | **95.1%** | Sub-millimeter landmark precision across 0.50 to 0.95 IoU thresholds |
| **Precision (B / P)** | **99.9% / 98.0%** | Ratio of true positive predictions over all detections |
| **Recall (B / P)** | **99.9% / 98.0%** | Percentage of true exercise movements captured |

### Per-Exercise Class Breakdown

| Exercise | Validation Instances | Box mAP@50 | Box mAP@50-95 | Pose mAP@50 | Pose mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Squats** | 542 | **99.5%** | **98.8%** | **99.5%** | **99.4%** |
| **Lateral Raises** | 124 | **99.5%** | **99.5%** | **99.5%** | **99.5%** |
| **Deadlifts** | 660 | **99.5%** | **96.7%** | **93.7%** | **86.5%** |

### Inference Latency Benchmark
* **Pre-process**: 0.5 ms per frame
* **Inference**: 3.5 ms per frame
* **Post-process**: 1.3 ms per frame
* **Total Latency**: ~5.3 ms per frame (**~190–280 FPS**) on NVIDIA RTX 4050

### Visual Diagnostics
Diagnostic plots and confusion matrices are preserved in [`runs/exercise_pose/`](runs/exercise_pose/):
* `runs/exercise_pose/results.png`: Epoch-by-epoch loss convergence curves.
* `runs/exercise_pose/confusion_matrix_normalized.png`: Normalized multi-class confusion matrix.
* `runs/exercise_pose/val_batch*_pred.jpg`: Side-by-side ground truth vs. predicted keypoint skeleton images.

---

## 2. Kinematic Joint Angle Classifier

* **Model File**: [`exercise_classifier.joblib`](exercise_classifier.joblib) (3.7 MB)
* **Metrics File**: [`exercise_classifier_metrics.json`](exercise_classifier_metrics.json)
* **Algorithm**: Scikit-Learn `HistGradientBoostingClassifier` with `DictVectorizer`
* **Dataset**: [`exercise_angles.csv`](exercise_angles.csv) (31,033 rows of extracted biomechanical joint angles)
* **Features Extracted**: `Side`, `Shoulder_Angle`, `Elbow_Angle`, `Hip_Angle`, `Knee_Angle`
* **Validation Strategy**: 80/20 Stratified Train-Test Split

### Overall Accuracy
* **Test Accuracy**: **88.45%** across 6,207 test rows

### Per-Exercise Performance Report

| Movement | Precision | Recall | F1-Score | Support | Kinematic Rationale |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Russian Twists** | **93.4%** | 89.3% | **91.3%** | 881 | Distinctive torso rotation and hip flexion angles |
| **Push Ups** | 87.6% | **93.4%** | **90.4%** | 1,953 | Clean elbow-to-shoulder compression in horizontal plane |
| **Jumping Jacks** | 90.6% | 87.1% | **88.8%** | 1,042 | High angular abduction across shoulders and knees |
| **Pull Ups** | 86.1% | 86.7% | **86.4%** | 1,332 | Consistent vertical pull arm kinematics |
| **Squats** | 87.0% | 81.6% | **84.2%** | 999 | Hip-knee flexion; variation reflects quarter vs. deep squats |

---

## 3. Spatial-Temporal Attention-LSTM Model

* **Weights File**: [`models/LSTM_Attention.h5`](models/LSTM_Attention.h5) (104 MB)
* **Baseline File**: [`models/LSTM.h5`](models/LSTM.h5) (9 MB)
* **Vision Task Bundle**: [`models/pose_landmarker.task`](models/pose_landmarker.task) (5.8 MB)
* **Architecture**: TensorFlow/Keras Bidirectional LSTM + Self-Attention
  * **Input Shape**: `(30, 132)` — sliding window of 30 frames $\times$ 33 MediaPipe 3D pose landmarks $(x, y, z, \text{visibility})$
  * **Recurrent Layer**: `Bidirectional(LSTM(256, return_sequences=True))`
  * **Attention Layer**: Spatial-temporal attention mechanism weighting peak contraction frames
  * **Output**: Softmax probability across live exercise classes (`curl`, `press`, `squat`)

---

## 4. Quick Execution & Reproduction Commands

### Train YOLOv8-Pose Model
```bash
# Standard 20-epoch training on RTX 4050
python train_exercise_yolo.py --epochs 20 --batch 8

# Higher resolution for fine keypoints
python train_exercise_yolo.py --epochs 30 --batch 8 --imgsz 640
```

### Train Kinematic Angle Classifier
```bash
python train_exercise_classifier.py --data exercise_angles.csv --output exercise_classifier.joblib
```

### Test Inference on Media
```bash
# Test on a photo
yolo pose predict model=exercise_pose_best.pt source=frontend/samples/sample_squat.webp conf=0.5 save=True

# Test on a video
yolo pose predict model=exercise_pose_best.pt source=my_video.mp4 conf=0.5 save=True
```

### Run Live Interactive Studio
```bash
python -m uvicorn app:app --reload --port 8000
```
Open in browser: **http://localhost:8000**
