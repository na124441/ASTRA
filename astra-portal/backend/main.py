"""
============================================================================
OWNER: Backend Developer 2
PURPOSE: FastAPI Root Entrypoint & CORS configuration.

HOW TO EDIT:
1. Add new APIRouters under `app.include_router(...)`.
2. Configure production domain origins in CORS middleware.
============================================================================
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.models import router as models_router
from api.downloads import router as downloads_router
from api.checksum import router as checksum_router
from api.inference import router as inference_router
from api.telemetry import router as telemetry_router

app = FastAPI(
    title="ASTRA-E Portal API",
    description="Backend services for ASTRA-E Model Hub, Inference Demo, and Release Distribution",
    version="1.0.0",
)

# Enable CORS for Next.js development (localhost:3000) and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(models_router)
app.include_router(downloads_router)
app.include_router(checksum_router)
app.include_router(inference_router)
app.include_router(telemetry_router)


@app.get("/health")
def health_check():
    """Root health check endpoint."""
    return {"status": "healthy", "service": "ASTRA-E Portal API"}
