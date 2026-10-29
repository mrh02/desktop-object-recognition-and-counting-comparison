FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    YOLO_OFFLINE=true \
    YOLO_AUTOINSTALL=false \
    DO_NOT_TRACK=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7861 \
    APP_OPEN_BROWSER=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-docker.txt ./
# The CPU wheel index avoids downloading unused CUDA runtime packages.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -c requirements.txt \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.14.1+cpu torchvision==0.29.1+cpu \
    && python -m pip install --no-cache-dir -c requirements.txt -r requirements-docker.txt \
    && python -m pip check

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/outputs/app_runs \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser app.py vision_pipeline.py vlm_service.py ./
USER appuser

EXPOSE 7861
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7861/', timeout=4).close()"
CMD ["python", "app.py"]
