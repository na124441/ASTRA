"""
============================================================================
OWNER: Backend Developer 2
PURPOSE: System Health & Telemetry API (GET /api/v1/system/status).

HOW TO EDIT:
1. Add GPU memory checks (torch.cuda.is_available()).
2. Report connected camera nodes.
============================================================================
"""

from datetime import datetime
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/system", tags=["System Telemetry"])


@router.get("/status")
def get_system_status():
    """Report system health and inference cluster readiness."""
    return {
        "service": "ASTRA-E Ground & Station Gateway",
        "status": "operational",
        "mission": "Bhartiya Antariksh Station (BAS)",
        "problem_statement": "SIH 26174",
        "active_models": 3,
        "dataset_runs_synced": 128,
        "edge_nodes_online": 4,
        "last_updated": datetime.utcnow().isoformat() + "Z",
    }
