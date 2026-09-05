"""Unit tests verifying the ASTRA Collector Web App static delivery and routes."""

from fastapi.testclient import TestClient
from apps.upload_api.main import app


def test_webapp_routes_and_static_assets():
    client = TestClient(app)

    # 1. Root route serves index.html
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "ASTRA COLLECTOR" in root_resp.text
    assert "text/html" in root_resp.headers.get("content-type", "")

    # 2. /collector route serves index.html
    collector_resp = client.get("/collector")
    assert collector_resp.status_code == 200
    assert "screen-camera" in collector_resp.text

    # 3. Static CSS file is served
    css_resp = client.get("/static/style.css")
    assert css_resp.status_code == 200
    assert "--cyan-accent" in css_resp.text

    # 4. Static JS file is served
    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "IndexedDBManager" in js_resp.text
    assert "deleteVerifiedVideo" in js_resp.text

    # 5. PWA Manifest is served
    manifest_resp = client.get("/static/manifest.json")
    assert manifest_resp.status_code == 200
    assert "ASTRA Collector" in manifest_resp.text
