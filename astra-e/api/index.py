"""Vercel Serverless Function entrypoint for ASTRA Collector API with route normalization."""

import sys
from pathlib import Path

# Ensure project root is available on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastapi import Request
from apps.upload_api.main import app


@app.middleware("http")
async def vercel_routing_middleware(request: Request, call_next):
    """
    Ensure routes match properly when deployed on Vercel.
    Handles Vercel CLI 59+ internal rewrites and x-matched-path headers.
    """
    matched = request.headers.get("x-matched-path")
    if matched and matched not in ("/api/index.py", "/api/index", "/api"):
        request.scope["path"] = matched
    else:
        path = request.scope.get("path", "")
        if path.startswith("/api/index.py"):
            remainder = path[len("/api/index.py"):]
            request.scope["path"] = remainder if remainder else "/collector"
        elif path in ("/api", "/api/", "/api/index"):
            request.scope["path"] = "/collector"

    return await call_next(request)
