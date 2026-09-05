"""Vercel Serverless Function entrypoint for ASTRA Collector API with route normalization."""

import sys
from pathlib import Path

# Ensure project root is available on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from apps.upload_api.main import app as fastapi_app


class VercelPathNormalizer:
    """
    ASGI middleware ensuring routes are matched properly on Vercel.
    In Vercel CLI 59+, internal rewrites forward requests using the destination
    path (/api/index.py). This middleware extracts the client's original requested
    path from 'x-matched-path' or normalizes /api/index.py paths.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            # Vercel provides the client's actual URL in x-matched-path
            matched_path = headers.get(b"x-matched-path", b"").decode("utf-8")
            
            if matched_path:
                scope["path"] = matched_path
            else:
                path = scope.get("path", "")
                if path in ("/api/index.py", "/api", "/api/index"):
                    scope["path"] = "/collector"
                elif path.startswith("/api/index.py/"):
                    scope["path"] = path[len("/api/index.py"):]

        await self.app(scope, receive, send)


app = VercelPathNormalizer(fastapi_app)
