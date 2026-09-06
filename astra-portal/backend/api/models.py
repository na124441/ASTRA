"""
============================================================================
OWNER: Backend Developer 1
PURPOSE: Model Registry API Endpoints (GET /api/v1/models).

HOW TO EDIT:
1. Load model metadata dynamically or add new fine-tuned checkpoints.
2. Filter models by hardware target (e.g. ?target=edge or ?precision=int8).
============================================================================
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/models", tags=["Models"])

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "models_catalog.json"


def load_catalog():
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": []}


@router.get("")
def list_models():
    """List all available trained models and specifications."""
    return load_catalog()


@router.get("/{model_id}")
def get_model(model_id: str):
    """Get details for a specific model checkpoint."""
    catalog = load_catalog()
    for m in catalog.get("models", []):
        if m["id"] == model_id:
            return m
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
