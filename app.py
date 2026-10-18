"""FastAPI Application for AI Exercise Recognition & Rep Counting.

Serves 1,324+ exercises with GIF animations, instructions, categories,
equipment data, and real-time rep counting for 26+ exercise types.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from exercise_engine import ExerciseAI
from movement_schema import ACTION_FAMILY

load_dotenv(Path(__file__).with_name(".env"))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}


class SettingsUpdate(BaseModel):
    min_detection_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_tracking_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    classification_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class TargetUpdate(BaseModel):
    action: Optional[str] = None


# The registry tags exercises with generic angle heuristics, but the live engine
# only counts the movements below. Map registry keys onto engine actions so the
# library advertises rep counting only where the camera pipeline delivers it.
HEURISTIC_TO_ACTION = {
    "bicep_curl": "bicep_curl",
    "overhead_press": "shoulder_press",
    "push_up": "push_up",
    "bench_press": "bench_press",
    "pull_up": "pull_up",
    "lat_pulldown": "pull_up",
    "barbell_row": "row",
    "cable_row": "row",
    "squat": "squat",
    "lunge": "lunge",
    "deadlift": "deadlift",
    "jumping_jack": "jumping_jack",
}


def _load_exercise_registry() -> dict[str, Any]:
    """Load the unified exercise registry JSON (built by build_exercise_registry.py)."""
    registry_path = Path(__file__).with_name("exercise_registry.json")
    if not registry_path.is_file():
        return {"exercises": [], "heuristics": {}, "total_exercises": 0}
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    supported = 0
    for exercise in registry.get("exercises", []):
        rep = exercise.setdefault("rep_counting", {})
        action = HEURISTIC_TO_ACTION.get(rep.get("heuristic_key"))
        exercise["rep_action"] = action
        rep["supported"] = action is not None
        supported += action is not None
    registry["rep_counting_supported"] = supported
    return registry


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
    try:
        yield
    finally:
        if app.state.engine is not None:
            app.state.engine.close()


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


@app.middleware("http")
async def revalidate_frontend(request: Request, call_next):
    """Make browsers re-check the UI files so an updated app.js is never stale."""
    response = await call_next(request)
    path = request.url.path
    if not path.startswith(("/api/", "/exercise-media/")):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/v1/health", tags=["system"])
async def health(request: Request) -> Any:
    engine: ExerciseAI | None = request.app.state.engine
    registry = request.app.state.registry
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_loaded": False, "message": request.app.state.error},
        )
    return {
        "status": "ok",
        "model": engine.model_info,
        "model_loaded": engine.model_loaded,
        "actions": list(engine.ACTIONS),
        "rep_actions": list(engine.REP_ACTIONS),
        "family_map": engine.family_map,
        "equipment_detection": "not_trained",
        "experimental": True,
        "rep_counting": "experimental_heuristic_not_validated",
        "total_exercises": registry.get("total_exercises", 0),
        "rep_counting_supported": registry.get("rep_counting_supported", 0),
        "categories": list(registry.get("categories", {}).keys()),
        "equipment_types": list(registry.get("equipment_types", {}).keys()),
    }


@app.post("/api/v1/process_frame", tags=["exercise"])
async def process_frame(
    request: Request, file: UploadFile = File(...), timestamp_seconds: Optional[float] = Form(None)
) -> dict:
    engine: ExerciseAI | None = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI engine not loaded: " + str(request.app.state.error))
    if timestamp_seconds is not None and (not math.isfinite(timestamp_seconds) or timestamp_seconds < 0):
        raise HTTPException(status_code=422, detail="timestamp_seconds must be finite and nonnegative")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=422, detail="The uploaded frame is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Frame size exceeds 10 MB limit.")

    try:
        result = await run_in_threadpool(engine.process_frame, contents, timestamp_seconds)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "success", "data": result}


# Backward compatibility for analyze endpoint
@app.post("/api/v1/analyze", include_in_schema=False)
async def analyze_alias(
    request: Request, file: UploadFile = File(...), confidence: float = 0.35,
    timestamp_seconds: Optional[float] = Form(None),
) -> dict:
    return await process_frame(request, file=file, timestamp_seconds=timestamp_seconds)


@app.post("/api/v1/reset_counters", tags=["exercise"])
async def reset_counters(request: Request) -> dict:
    engine: ExerciseAI | None = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI engine unavailable")
    counters = await run_in_threadpool(engine.reset_counters)
    return {"status": "success", "message": "Workout counters reset", "counters": counters}


@app.post("/api/v1/settings", tags=["exercise"])
async def update_settings(request: Request, payload: SettingsUpdate) -> dict:
    engine: ExerciseAI | None = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI engine unavailable")
    await run_in_threadpool(
        engine.update_settings,
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


@app.post("/api/v1/target", tags=["exercise"])
async def set_target(request: Request, payload: TargetUpdate) -> dict:
    """Pick the exercise to count reps for, or send null to follow the classifier."""
    engine: ExerciseAI | None = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI engine unavailable")
    try:
        target = engine.set_target(payload.action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "success", "target_action": target, "rep_actions": list(engine.REP_ACTIONS)}


@app.get("/api/v1/stats", tags=["exercise"])
async def get_stats(request: Request) -> dict:
    engine: ExerciseAI | None = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="AI engine unavailable")
    snap = engine.snapshot()
    return {
        "status": "success",
        "current_action": snap["current_action"],
        "display_name": snap["current_action"].replace("_", " ").title(),
        "confidence": snap["confidence"],
        "counters": snap["counters"],
        "stages": snap["stages"],
        "probabilities": snap["probabilities"],
        "target_action": snap["target_action"],
        "family": ACTION_FAMILY.get(snap["current_action"], "unknown"),
        "equipment": "unknown",
        "equipment_detection": "not_trained",
        "experimental": True,
        "model_info": engine.model_info,
        "rep_counting": "experimental_heuristic_not_validated",
    }


@app.get("/api/v1/model/metrics", tags=["system"])
async def model_metrics() -> dict:
    """Held-out evaluation of the movement classifier, for the accuracy page."""
    path = Path(__file__).parent / "models" / "exercise_starter_v3_metrics.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Model metrics file not found")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    splits = {}
    for name, split in metrics.get("splits", {}).items():
        raw = split.get("raw", {})
        splits[name] = {
            "windows": split.get("windows"),
            "groups": split.get("group_count"),
            "accuracy": raw.get("accuracy"),
            "macro_f1": raw.get("macro_f1"),
            "f1": {label: values.get("f1-score") for label, values in raw.get("per_class", {}).items()},
            "support": split.get("class_support", {}),
        }
    return {
        "status": "success",
        "clips": metrics.get("clips"),
        "usable_clips": metrics.get("usable_clips"),
        "windows": metrics.get("windows"),
        "selected": metrics.get("selected"),
        "splits": splits,
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
            "rep_action": e.get("rep_action"),
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

# The React redesign (web/, built with `npm run build`) is the main interface.
# The original vanilla UI stays available at /classic/, and serves / when the
# React build has not been produced.
classic_dir = Path(__file__).with_name("frontend")
web_dist_dir = Path(__file__).parent / "web" / "dist"
if (web_dist_dir / "index.html").is_file():
    app.mount("/classic", StaticFiles(directory=classic_dir, html=True), name="classic")
    app.mount("/", StaticFiles(directory=web_dist_dir, html=True), name="web")
else:
    app.mount("/", StaticFiles(directory=classic_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8001")), reload=False)
