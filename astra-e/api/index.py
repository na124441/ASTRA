"""Vercel Serverless Function entrypoint for ASTRA Collector API."""

import sys
from pathlib import Path

# Ensure project root is available on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from apps.upload_api.main import app

# Vercel WSGI/ASGI handler
app = app
