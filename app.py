"""FastAPI Application for AI Exercise Recognition & Rep Counting.

Serves 1,324+ exercises with GIF animations, instructions, categories,
equipment data, and real-time rep counting for 26+ exercise types.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from exercise_engine import ExerciseAI

load_dotenv(Path(__file__).with_name(".env"))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}


class SettingsUpdate(BaseModel):
    min_detection_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_tracking_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    classification_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


def _load_exercise_registry() -> dict[str, Any]:
    """Load the unified exercise registry JSON (built by build_exercise_registry.py)."""
    registry_path = Path(__file__).with_name("exercise_registry.json")
    if not registry_path.is_file():
        return {"exercises": [], "heuristics": {}, "total_exercises": 0}
    return json.loads(registry_path.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load exercise registry
    app.state.registry = _load_exercise_registry()
    print(f"Exercise registry loaded: {app.state.registry.get('total_exercises', 0)} exercises")
    print(f"  Rep counting supported: {app.state.registry.get('rep_counting_supported', 0)} exercises")

    try:
        app.state.engine = ExerciseAI()
        app.state.error = None
        print("ExerciseAI engine initialized successfully.")
    except Exception as error:
        app.state.engine = None
        app.state.error = str(error)
        print(f"Error initializing ExerciseAI engine: {error}")
    yield


app = FastAPI(
    title="AI Personal Fitness Trainer & Exercise Recognition API",
    version="3.0.0",
    description=(
        "Real-time exercise recognition and repetition counting for 1,324+ gym exercises. "
        "Covers 10 body categories, 28 equipment types, and 19 target muscle groups."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", tags=["system"])
async def health(request: Request) -> dict:
    engine: ExerciseAI | None = request.app.state.engine
    registry = request.app.state.registry
    if engine is None:
        return {"status": "degraded", "message": request.app.state.error}
    return {
        "status": "ok",
        "model": "LSTM_Attention (MediaPipe Pose 33 Keypoints)",
        "model_loaded": engine.model_loaded,
        "actions": ["curl", "press", "squat"],
        "total_exercises": registry.get("total_exercises", 0),
        "rep_counting_supported": registry.get("rep_counting_supported", 0),
        "categories": list(registry.get("categories", {}).keys()),
        "equipment_types": list(registry.get("equipment_types", {}).keys()),
    }


@app.post("/api/v1/process_frame", tags=["exercise"])
async def process_frame(request: Request, file: UploadFile = File(...)) -> dict:
    engine: ExerciseAI | None = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI engine not loaded: " + str(request.app.state.error))

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=422, detail="The uploaded frame is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Frame size exceeds 10 MB limit.")

    result = engine.process_frame(contents)
    return {"status": "success", "data": result}


# Backward compatibility for analyze endpoint
@app.post("/api/v1/analyze", include_in_schema=False)
async def analyze_alias(request: Request, file: UploadFile = File(...), confidence: float = 0.35) -> dict:
    return await process_frame(request, file)


@app.post("/api/v1/reset_counters", tags=["exercise"])
async def reset_counters(request: Request) -> dict:
    engine: ExerciseAI | None = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI engine unavailable")
    counters = engine.reset_counters()
    return {"status": "success", "message": "Workout counters reset", "counters": counters}


@app.post("/api/v1/settings", tags=["exercise"])
async def update_settings(request: Request, payload: SettingsUpdate) -> dict:
    engine: ExerciseAI | None = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI engine unavailable")
    engine.update_settings(
        min_detection_confidence=payload.min_detection_confidence,
        min_tracking_confidence=payload.min_tracking_confidence,
        classification_threshold=payload.classification_threshold,
    )
    return {
        "status": "success",
        "settings": {
            "min_detection_confidence": engine.min_detection_confidence,
            "min_tracking_confidence": engine.min_tracking_confidence,
            "classification_threshold": engine.classification_threshold,
        },
    }


@app.get("/api/v1/stats", tags=["exercise"])
async def get_stats(request: Request) -> dict:
    engine: ExerciseAI | None = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI engine unavailable")
    return {
        "status": "success",
        "current_action": engine.current_action,
        "confidence": engine.current_confidence,
        "counters": engine.counters,
        "stages": engine.stages,
        "probabilities": engine.last_probabilities,
    }


# ── Exercise Registry API ────────────────────────────────────────────────────

@app.get("/api/v1/exercises", tags=["exercises"])
async def list_exercises(
    request: Request,
    category: Optional[str] = None,
    body_part: Optional[str] = None,
    equipment: Optional[str] = None,
    target: Optional[str] = None,
    search: Optional[str] = None,
    rep_counting: Optional[bool] = None,
    page: int = 1,
    per_page: int = 50,
) -> dict:
    """Browse and search the full exercise database (1,324+ exercises).

    Supports filtering by category, body_part, equipment, target muscle,
    free-text search, and whether rep counting is supported.
    """
    registry = request.app.state.registry
    exercises = registry.get("exercises", [])

    # Apply filters
    if category:
        exercises = [e for e in exercises if e.get("category", "").lower() == category.lower()]
    if body_part:
        exercises = [e for e in exercises if e.get("body_part", "").lower() == body_part.lower()]
    if equipment:
        exercises = [e for e in exercises if e.get("equipment", "").lower() == equipment.lower()]
    if target:
        exercises = [e for e in exercises if e.get("target", "").lower() == target.lower()]
    if search:
        query = search.lower()
        exercises = [e for e in exercises if query in e.get("name", "").lower()]
    if rep_counting is not None:
        exercises = [e for e in exercises if e.get("rep_counting", {}).get("supported") == rep_counting]

    # Pagination
    total = len(exercises)
    start = (page - 1) * per_page
    end = start + per_page
    page_exercises = exercises[start:end]

    # Return lightweight summaries (no instructions to keep response small)
    summaries = [
        {
            "id": e["id"],
            "name": e["name"],
            "category": e["category"],
            "body_part": e["body_part"],
            "equipment": e["equipment"],
            "target": e["target"],
            "image": e.get("image", ""),
            "gif_url": e.get("gif_url", ""),
            "rep_counting_supported": e.get("rep_counting", {}).get("supported", False),
        }
        for e in page_exercises
    ]

    return {
        "status": "success",
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if per_page else 1,
        "exercises": summaries,
    }


@app.get("/api/v1/exercises/{exercise_id}", tags=["exercises"])
async def get_exercise(request: Request, exercise_id: str) -> dict:
    """Get full details for a single exercise including instructions, GIF, and rep heuristics."""
    registry = request.app.state.registry
    exercises = registry.get("exercises", [])

    exercise = next((e for e in exercises if str(e.get("id", "")) == exercise_id), None)
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")

    # Include heuristic details if rep counting is supported
    heuristic_detail = None
    heuristic_key = exercise.get("rep_counting", {}).get("heuristic_key")
    if heuristic_key:
        heuristic_detail = registry.get("heuristics", {}).get(heuristic_key)

    return {
        "status": "success",
        "exercise": {
            **exercise,
            "rep_heuristic": heuristic_detail,
        },
    }


@app.get("/api/v1/exercises/categories/summary", tags=["exercises"])
async def exercise_categories(request: Request) -> dict:
    """Get all exercise categories, equipment, and target muscles with counts."""
    registry = request.app.state.registry
    if "targets" not in registry:
        targets_count: dict[str, int] = {}
        for ex in registry.get("exercises", []):
            tgt = ex.get("target")
            if tgt:
                targets_count[tgt] = targets_count.get(tgt, 0) + 1
        registry["targets"] = dict(sorted(targets_count.items(), key=lambda x: x[0]))

    return {
        "status": "success",
        "categories": registry.get("categories", {}),
        "equipment_types": registry.get("equipment_types", {}),
        "targets": registry.get("targets", {}),
    }


# ── Serve exercise media from exercises-dataset ──────────────────────────────
exercises_dataset_dir = Path(__file__).with_name("exercises-dataset")
if exercises_dataset_dir.is_dir():
    app.mount(
        "/exercise-media",
        StaticFiles(directory=exercises_dataset_dir),
        name="exercise_media",
    )

app.mount("/", StaticFiles(directory=Path(__file__).with_name("frontend"), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=False)
