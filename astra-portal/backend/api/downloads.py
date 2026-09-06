"""
============================================================================
OWNER: Backend Developer 1
PURPOSE: Artifact Download Streamer / Proxy to private Hugging Face repo.

HOW TO EDIT:
1. Add streaming download support with Range headers for large weights.
2. Implement signed temporary URLs if models require strict access control.
============================================================================
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/api/v1/downloads", tags=["Downloads"])

HF_BASE_URL = "https://huggingface.co/na124441/astra-e-raw/resolve/main/models"


@router.get("/{model_id}")
def download_model(model_id: str):
    """
    Redirect to authentic Hugging Face release asset with authorization token
    if required.
    """
    valid_ids = {
        "astra-exp001-int8": "exp001-int8.onnx",
        "astra-exp001-fp16": "exp001-fp16.pt",
        "astra-exp001-tensorrt": "exp001-tensorrt.engine",
    }

    if model_id not in valid_ids:
        raise HTTPException(status_code=404, detail="Requested model artifact not found.")

    filename = valid_ids[model_id]
    target_url = f"{HF_BASE_URL}/{filename}"

    # If HF_TOKEN is configured in environment, append as Bearer or query redirect
    return RedirectResponse(url=target_url, status_code=307)
