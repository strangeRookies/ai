FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    ULTRALYTICS_CONFIG_DIR=/tmp/ultralytics \
    YOLO_MODEL_PATH=/models/yolo26n-pose.pt \
    MODEL_CHECKPOINT_PATH=/models/lstm.pt \
    VIDEO_POOL_DIR=/data/videos \
    EVENT_CLIP_OUTPUT_DIR=/app/runs/clips \
    DEVICE=auto

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY ai ./ai
COPY detector ./detector
COPY messaging ./messaging
COPY rules ./rules
COPY scripts ./scripts
COPY stream ./stream
COPY tracking ./tracking
COPY config.py main.py ./

RUN mkdir -p /models /data/videos /app/runs /tmp/matplotlib /tmp/ultralytics \
    && adduser --system --group aiuser \
    && chown -R aiuser:aiuser /app/runs /tmp/matplotlib /tmp/ultralytics

VOLUME ["/models", "/data/videos", "/app/runs"]

USER aiuser

CMD ["python", "scripts/run_registered_cameras.py"]
