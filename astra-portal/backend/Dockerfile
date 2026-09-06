# ============================================================================
# Dockerfile: ASTRA-E Portal FastAPI Backend Service
# Targeted Deployment Platforms: Hugging Face Spaces / Render / Docker Hub
# ============================================================================

FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing pyc files to disk and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose port 8000 (Hugging Face Spaces default is 7860 or 8000)
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
