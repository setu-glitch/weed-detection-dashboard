# Container image for the weed-detection dashboard.
# Works on Google Cloud Run, Fly.io, Koyeb, a plain VM, or anywhere that runs
# Docker. Built for CPU inference; no CUDA runtime is installed.

FROM python:3.11-slim

# Ultralytics imports OpenCV, which needs these two system libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp \
    MPLCONFIGDIR=/tmp

WORKDIR /app

# Dependencies first, so code changes do not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run and several other hosts inject the port to listen on.
ENV PORT=8501
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8501)}/_stcore/health')" || exit 1

CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
