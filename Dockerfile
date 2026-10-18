# FitnessCoach production image (used by Render; runs anywhere Docker does).
#
#   docker build -t fitnesscoach .
#   docker run -p 8000:8000 fitnesscoach
#
# Stage 1 builds the React interface; stage 2 is the FastAPI server that
# serves it together with the API, the classic UI and the exercise library.

FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# OpenCV (pulled in by MediaPipe) needs libGL/glib, MediaPipe needs EGL/GLES, it imports
# sounddevice, which needs PortAudio; git fetches the exercise dataset.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libegl1 libgles2 libglib2.0-0 libportaudio2 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# TensorFlow is deliberately not installed: MediaPipe only imports it when
# present, and leaving it out keeps the server inside small memory limits.
COPY requirements-lock.txt ./
RUN pip install -r requirements-lock.txt

# The exercise library is not stored in this repository (see .gitignore).
# Fetch the public dataset at a pinned commit and build the registry from it.
ARG EXERCISES_REPO=https://github.com/hasaneyldrm/exercises-dataset.git
ARG EXERCISES_REF=7455efa
RUN git clone --quiet "$EXERCISES_REPO" exercises-dataset \
    && git -C exercises-dataset checkout --quiet "$EXERCISES_REF" \
    && rm -rf exercises-dataset/.git

COPY *.py ./
COPY models/ models/
COPY frontend/ frontend/
COPY --from=web /web/dist web/dist
RUN python build_exercise_registry.py

# Render sets $PORT; default to 8000 elsewhere.
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
