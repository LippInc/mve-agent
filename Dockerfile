# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Layer-cache-friendly: requirements first, code second
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/

# Config is env-injected at run time (FIREWORKS_API_KEY, FIREWORKS_BASE_URL,
# ALLOWED_MODELS). Nothing is bundled, nothing is downloaded at runtime.
ENTRYPOINT ["python", "-m", "agent.main"]
