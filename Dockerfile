# ── Hugging Face Spaces — Docker Runtime ─────────────────────────────────────
# Python 3.11 slim keeps the image lean while supporting all crewai deps.
FROM python:3.11.9-slim

# Prevent .pyc files and force unbuffered stdout/stderr (crucial for HF logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── System deps required by crewai / grpc / tiktoken ─────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (cached layer — only rebuilds when requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application files ─────────────────────────────────────────────────────────
COPY app.py .
COPY index.html .

# ── Memory file: pre-create so the container can write to it immediately ──────
# On HF Spaces the filesystem is ephemeral, but this ensures the file exists
# on first boot without any code-level FileNotFoundError.
RUN echo "[]" > /app/memory.json

# ── Hugging Face Spaces REQUIRES port 7860 ────────────────────────────────────
ENV PORT=7860
EXPOSE 7860

# ── Gunicorn config ───────────────────────────────────────────────────────────
# 1 worker + 8 threads  → ideal for HF single-instance Spaces
# timeout 300           → multi-agent CrewAI runs can take 2-4 min
# graceful-timeout 10   → clean shutdown on redeploy
CMD exec gunicorn \
    --bind "0.0.0.0:7860" \
    --workers 1 \
    --threads 8 \
    --timeout 300 \
    --graceful-timeout 10 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    app:app
