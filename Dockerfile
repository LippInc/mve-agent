# syntax=docker/dockerfile:1
# Pin the platform so the graded linux/amd64 manifest is guaranteed regardless
# of the build host or invocation (belt-and-suspenders with buildx --platform).
FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

# Layer-cache-friendly: requirements first, code second
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/

# Config is env-injected at run time (FIREWORKS_API_KEY, FIREWORKS_BASE_URL,
# ALLOWED_MODELS). Nothing is bundled, nothing is downloaded at runtime.
ENTRYPOINT ["python", "-m", "agent.main"]
