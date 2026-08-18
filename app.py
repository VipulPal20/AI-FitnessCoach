"""FastAPI Application for AI Exercise Recognition & Rep Counting."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

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


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    version="2.0.0",
    description="Real-time exercise recognition (Curls, Press, Squats) and repetition counting.",
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
    if engine is None:
        return {"status": "degraded", "message": request.app.state.error}
    return {
        "status": "ok",
        "model": "LSTM_Attention (MediaPipe Pose 33 Keypoints)",
        "model_loaded": engine.model_loaded,
        "actions": ["curl", "press", "squat"],
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


app.mount("/", StaticFiles(directory=Path(__file__).with_name("frontend"), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=False)
