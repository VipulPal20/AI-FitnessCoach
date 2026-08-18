"""FastAPI service for image-based exercise form analysis."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from vision import InvalidImageError, PoseAnalyzer
from smartapi_service import SmartAPIUnavailable, SmartMarketService
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).with_name(".env"))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.analyzer = PoseAnalyzer(os.getenv("POSE_MODEL_PATH"))
        app.state.model_error = None
    # The market dashboard and static frontend remain usable even if the optional
    # local pose-model runtime cannot start in the host environment.
    except Exception as error:
        app.state.analyzer = None
        app.state.model_error = str(error)
    app.state.market = SmartMarketService()
    yield


app = FastAPI(
    title="Form Coach Vision API",
    version="1.0.0",
    description="Upload an image to receive pose landmarks and basic squat-form feedback.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(InvalidImageError)
async def invalid_image_handler(_: Request, error: InvalidImageError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"status": "error", "message": str(error)})


@app.get("/api/v1/health", tags=["service"])
async def health(request: Request) -> dict[str, str]:
    analyzer: PoseAnalyzer | None = request.app.state.analyzer
    if analyzer is None:
        return {"status": "degraded", "model": "unavailable", "message": request.app.state.model_error}
    return {"status": "ok", "model": analyzer.model_path.name}




@app.post("/api/v1/analyze", tags=["form analysis"])
async def analyse_image(request: Request, file: UploadFile = File(...), confidence: float = 0.35) -> dict:
    """Analyse a JPEG, PNG, or WebP upload. `confidence` must be between 0.1 and 0.9."""
    if not 0.1 <= confidence <= 0.9:
        raise HTTPException(status_code=422, detail="confidence must be between 0.1 and 0.9")
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    analyzer: PoseAnalyzer | None = request.app.state.analyzer
    if analyzer is None:
        raise HTTPException(status_code=503, detail=request.app.state.model_error)
    return {"status": "success", "data": analyzer.process_frame(contents, confidence)}


# Compatibility alias for the original client contract.
@app.post("/predict", include_in_schema=False)
async def predict(request: Request, file: UploadFile = File(...), confidence: float = 0.35) -> dict:
    return await analyse_image(request, file, confidence)


app.mount("/", StaticFiles(directory=Path(__file__).with_name("frontend"), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
