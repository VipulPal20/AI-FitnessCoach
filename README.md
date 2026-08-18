# Form Coach Vision API

FastAPI backend that detects a person's pose in an uploaded image and produces practical squat-form feedback.

The bundled YOLOv8 pose model runs locally. When `exercise_pose_best.pt` exists, the app automatically uses the fine-tuned exercise-pose model to detect `deadlift`, `lateral_raise`, and `squat`; otherwise it falls back to the general pose model. The app automatically keeps Ultralytics and Matplotlib cache files in the project, so it also starts on managed machines where the normal user-profile cache directories are not writable.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://localhost:8000/docs` to try the API.

## API

`POST /api/v1/analyze` accepts a multipart `file` (`image/jpeg`, `image/png`, or `image/webp`) and optional `confidence` between `0.1` and `0.9`.
It returns the strongest detected pose plus squat depth, left/right knee angles, knee-angle symmetry, torso lean, knee-tracking guidance, visible-keypoint count, and actionable feedback. These estimates are coaching aids, not a medical assessment.

The included `exercise_classifier.joblib` is trained from `exercise_angles.csv` to classify Jumping Jacks, Pull ups, Push Ups, Russian twists, and Squats from pose-derived angles. Its prediction appears at `people[0].metrics.exercise` when enough landmarks are visible. Low-confidence predictions are returned as `label: "unknown"`, along with a candidate label, rather than presented as a reliable result. Re-train it after updating the dataset:

```powershell
python train_exercise_classifier.py
```

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/analyze" -F "file=@squat.webp"
```

`GET /api/v1/health` confirms the service and loaded model. The legacy `POST /predict` endpoint remains available.

## Configuration

- `POSE_MODEL_PATH`: alternate path to a YOLO pose model.
- `EXERCISE_POSE_MODEL_PATH`: alternate path to the fine-tuned exercise YOLO-pose model.
- `CORS_ORIGINS`: comma-separated allowed origins. Defaults to `*` for local development.
- `YOLO_CONFIG_DIR`: optional writable directory for Ultralytics settings. The app defaults to `.ultralytics` in this project.

## YOLO exercise-pose training

`archive (3).zip` was normalized into `exercise_pose_dataset` (3,270 training and 1,326 validation images). It contains pose labels for deadlifts, lateral raises, and squats. Run a full fine-tuning job where a GPU is available:

```powershell
python train_exercise_yolo.py --epochs 20 --imgsz 512 --batch 16
```

## SmartAPI market signals

The market dashboard is available at `http://localhost:8000/markets.html`. Copy `.env.example` to `.env` and set your Angel One SmartAPI credentials: `SMART_API_KEY`, `SMART_CLIENT_CODE`, `SMART_PIN`, and `SMART_TOTP_SECRET`. Credentials are used only by the server; never add them to browser code or commit `.env`.

The dashboard uses SmartAPI instrument search, live LTP, positions, and recent 5-minute candles to generate an educational technical signal. It does not place orders and is not investment advice.
