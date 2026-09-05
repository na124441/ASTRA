# Production Dockerfile for ASTRA Collector Upload API
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_RAW_DATASET_REPO="na124441/astra-e-raw"

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /workspace/
RUN pip install --no-cache-dir fastapi uvicorn pydantic huggingface_hub httpx aiofiles

COPY apps/ /workspace/apps/
COPY astra/ /workspace/astra/

EXPOSE 8000

CMD ["uvicorn", "apps.upload_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
