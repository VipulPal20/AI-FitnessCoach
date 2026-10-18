<p align="center">
  <img src="frontend/assets/fitnesscoach-logo.svg" alt="FitnessCoach Logo" width="460">
</p>

<p align="center">
  <strong>AI Based Fitness Evaluator & Real-Time Exercise Trainer Engine</strong>
</p>

An AI-powered computer vision application for real-time exercise classification, pose estimation, joint kinematic analysis, and repetition counting, with a browsable library of 1,324 gym exercises. Built with **FastAPI**, **MediaPipe Pose**, a fitted **scikit-learn Decision Tree**, and **OpenCV**, featuring an interactive web dashboard.

---

## ✨ Key Features

- 🏋️ **Experimental Real-Time Exercise Recognition**: Uses a hierarchical v3 pose model over an approximately 1.9-second temporal window for 11 canonical movement labels.
- 🧠 **Fitted Decision-Tree Model**: Interpretable hierarchical classifier trained on normalized 33-point MediaPipe pose sequences with grouped train/validation/test splits.
- 🔢 **Experimental Repetition Counter**: Joint-angle state machine (`START` → `EFFORT` → `START` = one rep) for 10 movements. Reps can follow the AI's guess, or a target exercise you pick for reliable counting.
- 📐 **Kinematic Joint Angle & Form Analysis**: Calculates elbow, shoulder, hip, and knee angles in real time, with on-screen coaching cues.
- 📚 **Exercise Library**: 1,324 exercises with images, GIF demonstrations, instructions, and filters by category, equipment, and target muscle; 429 of them map to a movement the live engine can count.
- 💻 **Interactive Web Dashboard**: Live webcam, video upload, photo analysis, a one-click squat demo video, skeleton overlay, and real-time stats.
- ⚙️ **Configurable AI Parameters**: Tune detection confidence, tracking confidence, and classification thresholds on the fly.
- ⚡ **RESTful API**: FastAPI endpoints for frame processing, rep targets, stats, counter resets, the exercise library, and health checks.

---

## 🏗️ Architecture

```
┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│  Camera / Video │ ────> │ MediaPipe Pose Task  │ ────> │ 33 3D Pose Keypoints   │
└─────────────────┘       └──────────────────────┘       └───────────┬────────────┘
                                                                      │ (normalized temporal features)
                                                                      ▼
┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│ Interactive UI  │ <──── │ Rep Counter Engine   │ <──── │ Fitted Decision Tree   │
└─────────────────┘       └──────────────────────┘       └────────────────────────┘
```

Each frame is posted to the server with its capture time. The classifier reads the last 1.9 seconds of poses; the rep counter reads the joint angles of the current frame.

---

## 📁 Repository Structure

```
AI-FitnessCoach/
├── app.py                         # FastAPI server & REST API endpoints
├── exercise_engine.py             # Core AI engine (MediaPipe, v3 classifier, rep tracking)
├── repetition_counter_v3.py       # Joint-angle rep state machine
├── pose_features_v3.py            # 1.9 s temporal pose features
├── movement_schema.py             # The 11 movement labels and their families
├── train_pose_v3.py               # v3 model training (also used for inference)
├── smoke_test_v3.py               # Sends a local video through the live API
├── start_fitnesscoach.bat         # One-click Windows launcher
├── requirements.txt               # Runtime dependencies (version ranges)
├── requirements-lock.txt          # Exact tested versions (recommended)
├── requirements-train.txt         # Extra dependencies for data prep & training
├── exercise_registry.json         # 1,324-exercise library (built by build_exercise_registry.py)
├── exercises-dataset/             # Exercise images, GIFs and instructions
├── models/
│   ├── exercise_starter_v3.joblib # Fitted v3 model used by the live engine
│   ├── exercise_starter_v3_metrics.json
│   └── pose_landmarker.task       # MediaPipe Pose Landmarker task binary
├── web/                           # React interface (Vite); `npm run build` -> web/dist, served at /
└── frontend/                      # Classic interface, served at /classic/ (or / without a React build)
    ├── index.html                 # Dashboard layout
    ├── styles.css                 # Styling
    ├── app.js                     # API client, video handling & skeleton overlay
    └── assets/ & samples/         # Logo, sample photos & squat demo video
```

---

## ⚡ Quick Start (Windows)

### 1. Prerequisites
- **Python 3.12** (64-bit), from [python.org](https://www.python.org/downloads/)
- A webcam (optional; needed only for Live Camera mode)

### 2. Installation

Open PowerShell in the project folder:

```powershell
# Create the virtual environment
py -3.12 -m venv .venv

# Install the exact versions the app was tested with
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The exercise library is not stored in this repository (`exercises-dataset/` and `exercise_registry.json` are git-ignored). On a fresh clone, fetch and build it once:

```powershell
git clone https://github.com/hasaneyldrm/exercises-dataset.git
.\.venv\Scripts\python.exe build_exercise_registry.py
```

Without it, the app still runs, but the Exercise Library tab will be empty.

`requirements.txt` holds the same packages as version ranges. Use it if a pinned version is not available for your Python. Training scripts need `requirements-train.txt` too.

### 3. Build the web interface (optional, recommended)

The main interface is a React app in `web/`. Building it needs [Node.js](https://nodejs.org/) 18 or newer:

```powershell
cd web
npm install
npm run build
cd ..
```

This creates `web/dist/`, which the server then serves at `/`. Without the build, `/` shows the classic interface instead, so the app still works without Node.js.

### 4. Running the Application

**Easiest:** double-click `start_fitnesscoach.bat`. It starts the server in its own console window, waits until the AI engine has loaded (about 30 seconds the first time), and then opens the dashboard. Close the "FitnessCoach Server" window to stop the app.

**Manually:**

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Then open:
- 🌐 **Web Dashboard**: [http://127.0.0.1:8001/](http://127.0.0.1:8001/) (the React interface if built, otherwise the classic one)
- 🧰 **Classic Dashboard**: [http://127.0.0.1:8001/classic/](http://127.0.0.1:8001/classic/) (available whenever the React build is present)
- 📖 **Interactive API Docs**: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

The status pill in the top bar shows **Experimental AI** once the model has loaded, or **Degraded** if it failed (the server window shows why).

---

## 🎮 Using the Classic Dashboard

The section below describes the classic interface in `frontend/`. The React interface at `/` talks to the same API.

| Mode | What it does |
| :--- | :--- |
| **Live Camera Studio** | Streams your webcam to the AI. Stand 2–3 m back with your whole body (or at least one full side) in frame. |
| **Upload Video / Photo** | A **video** is analysed while it plays, with reps counted and a summary shown at the end. A **photo** shows the skeleton and joint angles only, because naming an exercise needs about 2 seconds of movement. |
| **Video Analysis** | Video-only upload. **▶ Squat Video (rep demo)** plays a built-in 16-second clip with 2 complete squats. |
| **Exercise Library** | Search 1,324 exercises. **▶ Train** (or *Train This Exercise Live* / *Test Video* in the detail view) sets that exercise as the rep target. |

### Auto-detect vs. "Count reps for"

The **Count reps for** menu in the Current Exercise card controls rep counting:

- **Auto-detect (AI classifier)**: reps follow the exercise the model recognises. Because the model is experimental (see below), it can miss reps when it confuses similar movements.
- **A specific exercise** (Squat, Deadlift, Bicep Curl, …): reps are counted from joint angles for that exercise, whatever the model guesses. This mode is the most reliable for a demo. The model's own guess still appears in the title (e.g. `SQUAT · AI SEES LUNGE`) and in the Motion Classifier panel.

The phase chip shows `START` (the rest position) and `EFFORT` (the working position). One rep = START → EFFORT → START.

### Command-line smoke test

With the server running, any local video can be sent through the API:

```powershell
.\.venv\Scripts\python.exe smoke_test_v3.py path\to\workout.mp4
```

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Server health, model info, supported actions |
| `POST` | `/api/v1/process_frame` | Multipart `file` (JPEG/PNG/WebP frame) plus optional `timestamp_seconds`; returns pose landmarks, angles, classification, and rep state |
| `POST` | `/api/v1/target` | JSON `{"action": "squat"}` counts reps for that exercise; `{"action": null}` returns to auto-detect |
| `GET` | `/api/v1/model/metrics` | Held-out accuracy, macro-F1 and per-class F1 for each data split |
| `GET` | `/api/v1/stats` | Current exercise, confidence, rep counts, stages, rep target |
| `POST` | `/api/v1/reset_counters` | Resets all rep counters to `0` |
| `POST` | `/api/v1/settings` | Updates detection/tracking confidence and classification threshold |
| `GET` | `/api/v1/exercises` | Library search with `search`, `category`, `equipment`, `target`, `rep_counting`, `page`, `per_page` |
| `GET` | `/api/v1/exercises/{id}` | Full details for one exercise |
| `GET` | `/api/v1/exercises/categories/summary` | Categories, equipment and target muscles with counts |

Send frames with real, increasing `timestamp_seconds` (e.g. video time). A jump backwards or a gap over 2 s starts a fresh tracking window.

---

## 🧠 v3 Model Details

- **Input Features**: 792 normalized temporal features from 20 MediaPipe pose samples over 1.9 seconds.
- **Estimator**: Hierarchical `DecisionTreeClassifier`; the selected artifact uses `max_depth=8, min_samples_leaf=8`.
- **Output Classes**: `bench_press`, `shoulder_press`, `push_up`, `squat`, `deadlift`, `bicep_curl`, `pull_up`, `row`, `lunge`, `jumping_jack`, and `unknown`.
- **Stability**: A prediction must hold for 3 frames before it is adopted, and for 6 frames before it replaces an established exercise. Uncertain frames keep the last exercise. If the label flips between close relatives (squat ↔ lunge) in the middle of a rep, the rep continues and is credited to the label seen most often.
- **Equipment**: Not trained; the API reports `equipment_detection: "not_trained"`.

### Validation status

The current v3 artifact is **experimental, not production-ready**. It has complete grouped coverage for all 11 labels, but the source recordings do not provide verified person identities and the dataset grouping is only an approximate recording-level split. Held-out metrics from `models/exercise_starter_v3_metrics.json`:

| Split | Windows | Accuracy | Macro-F1 |
| :--- | ---: | ---: | ---: |
| Validation | 55 | 0.418 | 0.382 |
| Test | 74 | 0.554 | 0.462 |

Do not interpret these numbers as person-disjoint generalization or as validated rep-counting accuracy.

### Known limitations

- **Small training set**: 120 usable clips (360 windows). Some classes are often confused: on validation, 14 of 16 squat windows were predicted as lunge, and on test, 10 of 13 pull-up windows were predicted as shoulder press.
- **2D angle heuristics**: Rep thresholds use joint angles in the image plane, so camera angle matters. The deadlift threshold is set for a front-on view, where a full hip hinge only reads about 140°. Side-on footage gives the most dependable counts for most exercises.
- **Rep counting is not validated** against labelled rep counts. Spot checks on recorded clips with the exercise chosen under "Count reps for": 2/2 squats and 1/1 deadlift counted, 0 false reps on a lateral-raise clip. The squat count held at every sampling rate tried, from 10 frames/s down to 1 frame/s.
- **Single user**: The server keeps one shared tracking state, so use one browser tab at a time.
- **Movements outside the 10 counted exercises** (e.g. lateral raises, dips) are shown as form tracking only.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Vipul Pal.
