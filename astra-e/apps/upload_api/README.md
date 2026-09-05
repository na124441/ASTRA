# ASTRA-E Upload API Service

Authenticated microservice for receiving experimental procedure recordings from ASTRA Collector mobile clients and committing them to private Hugging Face Dataset repositories (`astra-e-raw`).

## Quickstart

### 1. Environment Setup
```bash
cp apps/upload_api/.env.example apps/upload_api/.env
```
Edit `.env` to supply your Hugging Face write token:
```env
HF_TOKEN=hf_...
HF_RAW_DATASET_REPO=na124441/astra-e-raw
MOCK_HF_UPLOAD=false
PORT=8000
```

### 2. Launch Local Development Server
```bash
python -m uvicorn apps.upload_api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run Automated Test Suite
```bash
python -m pytest tests/unit/upload_api tests/integration/upload_api -v
```

### 4. Docker Deployment
```bash
docker build -t astra-collector-api -f apps/upload_api/Dockerfile .
docker run -p 8000:8000 --env-file apps/upload_api/.env astra-collector-api
```
