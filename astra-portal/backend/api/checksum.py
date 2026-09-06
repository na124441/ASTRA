"""
============================================================================
OWNER: Backend Developer 1
PURPOSE: Checksum Verification Service (POST /api/v1/verify-checksum).

HOW TO EDIT:
1. Compares the client's computed file hash with our recorded golden hash.
2. Logs verification requests for integrity telemetry.
============================================================================
"""

import json
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/verify-checksum", tags=["Integrity"])

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "models_catalog.json"


class VerifyRequest(BaseModel):
    model_id: str
    client_sha256: str


class VerifyResponse(BaseModel):
    model_id: str
    is_authentic: bool
    expected_sha256: str
    status: str


@router.post("", response_model=VerifyResponse)
def verify_checksum(req: VerifyRequest):
    """Verify that a downloaded model matches the official release hash."""
    if not CATALOG_PATH.exists():
        raise HTTPException(status_code=500, detail="Catalog database unavailable.")

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    for m in catalog.get("models", []):
        if m["id"] == req.model_id:
            expected = m["sha256"].lower()
            provided = req.client_sha256.lower().strip()
            matches = expected == provided
            return VerifyResponse(
                model_id=req.model_id,
                is_authentic=matches,
                expected_sha256=expected,
                status="VERIFIED_AUTHENTIC" if matches else "HASH_MISMATCH",
            )

    raise HTTPException(status_code=404, detail=f"Model ID '{req.model_id}' unknown.")
