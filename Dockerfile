# Ava, the Sunrise Dental voice receptionist.
#
# Build:  docker build -t ava-voice-agent .
# Run:    docker run -p 7860:7860 --env-file .env -v ava-models:/models -v ava-data:/app/data ava-voice-agent
#
# The clinic documents (data/raw) are inside the image. If data/processed holds a search index made by
# scripts/ingest.py it is used; if not, the app builds the index itself when it starts.
#
# Size: an estimated 2-3 GB, mostly PyTorch and Transformers (not yet measured). The models themselves
# (about 1.2 GB) are NOT baked in; they download on first start into the /models volume,
# so later starts are fast and the image stays smaller.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

# Install the CPU-only build of PyTorch first (the default build includes GPU libraries and is several GB bigger).
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3" \
 && pip install -r requirements.txt

COPY . .

# Run as a normal user, not root, so a bug in the app cannot change the container's system files.
RUN useradd --create-home ava \
 && mkdir -p /app/data /app/logs /models \
 && chown -R ava /app /models
USER ava

EXPOSE 7860
# Docker marks the container "unhealthy" if /health stops answering. Models take a minute to load at first start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"

CMD ["python", "app.py"]
