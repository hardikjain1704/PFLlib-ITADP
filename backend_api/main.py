"""
Privacy Compliance API for PFLlib Federated Learning.

This backend proxies requests to the real federated aggregation server
(system/federated/server.py) running on port 9000.

For consent and purpose endpoints, it calls the aggregation server directly.
For training control, it also proxies to the aggregation server.

Run with:
    cd backend_api
    uvicorn main:app --reload --port 8000
"""

import os
import json
import requests
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# ── App setup ──────────────────────────────────────────────────
app = FastAPI(
    title="PFLlib Privacy Compliance API",
    description="Consent, purpose-limitation, and transparency endpoints. Proxies to the federated aggregation server.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Aggregation server URL ────────────────────────────────────
AGG_SERVER = os.environ.get("AGG_SERVER_URL", "http://127.0.0.1:9001")


def _agg_get(path: str):
    """GET from the aggregation server."""
    try:
        resp = requests.get(f"{AGG_SERVER}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(503, "Aggregation server not reachable. Start it with: python run_server.py")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(e.response.status_code, e.response.text)


def _agg_post(path: str, json_data=None, params=None):
    """POST to the aggregation server."""
    try:
        resp = requests.post(f"{AGG_SERVER}{path}", json=json_data, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(503, "Aggregation server not reachable. Start it with: python run_server.py")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(e.response.status_code, e.response.text)


# ── Request schemas ────────────────────────────────────────────
class ConsentRequest(BaseModel):
    client_id: int
    consent: bool
    purpose: str = "image_classification"


class PurposeValidationRequest(BaseModel):
    purpose: str
    dataset_features: List[str]


class StartTrainingRequest(BaseModel):
    global_rounds: int = 5
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    num_clients: int = 2


# ══════════════════════════════════════════════════════════════
#  CONSENT ENDPOINTS – proxy to aggregation server
# ══════════════════════════════════════════════════════════════

@app.post("/consent")
def post_consent(req: ConsentRequest):
    return _agg_post("/consent", json_data=req.model_dump())


@app.get("/consent/{client_id}")
def get_consent(client_id: int):
    return _agg_get(f"/consent/{client_id}")


@app.get("/consent")
def get_all_consent():
    return _agg_get("/consent")


# ══════════════════════════════════════════════════════════════
#  PURPOSE VALIDATION – proxy
# ══════════════════════════════════════════════════════════════

@app.post("/validate-purpose")
def validate_purpose(req: PurposeValidationRequest):
    return _agg_post("/validate-purpose", json_data=req.model_dump())


@app.get("/purpose-violations")
def get_purpose_violations():
    return _agg_get("/purpose-violations")


# ══════════════════════════════════════════════════════════════
#  TRANSPARENCY – proxy
# ══════════════════════════════════════════════════════════════

@app.get("/user-info/{user_id}")
def get_user_info(user_id: int):
    return _agg_get(f"/user-info/{user_id}")


@app.get("/transparency-log")
def get_transparency_log():
    return _agg_get("/transparency-log")


@app.get("/transparency-stats")
def get_transparency_stats():
    return _agg_get("/transparency-stats")


@app.get("/training-log")
def get_training_log():
    return _agg_get("/training-log")


# ══════════════════════════════════════════════════════════════
#  TRAINING CONTROL – proxy to federated server
# ══════════════════════════════════════════════════════════════

@app.post("/start-training")
def start_training(req: StartTrainingRequest):
    """Start a real federated training session via the aggregation server."""
    return _agg_post("/start-training", json_data=req.model_dump())


@app.get("/training-status")
def get_training_status():
    """Poll training status from the aggregation server."""
    return _agg_get("/training-status")


@app.post("/stop-training")
def stop_training():
    return _agg_post("/stop-training")


@app.get("/status")
def get_status():
    return _agg_get("/status")


# ══════════════════════════════════════════════════════════════
#  CLIENT INFO – proxy
# ══════════════════════════════════════════════════════════════

@app.get("/client/{client_id}")
def get_client_info(client_id: int):
    """Return client participation stats."""
    return _agg_get(f"/user-info/{client_id}")


# ── Health check ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "role": "backend_api_proxy"}
