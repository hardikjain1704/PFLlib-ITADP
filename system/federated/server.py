"""
Federated Aggregation Server.

Runs on Laptop A.  Accepts model-weight uploads from clients via HTTP,
performs FedAvg aggregation, and serves the global model.

Also exposes consent, purpose-validation, transparency, and training-log
endpoints so the dashboard can connect directly.

Endpoints:
    POST /upload-weights    — client sends local weights after training
    GET  /global-model      — client fetches the latest global model
    POST /start-round       — trigger a new training round
    GET  /round-config      — clients poll this for current config
    GET  /status            — full training status for dashboard
    POST /consent           — store consent with purpose
    GET  /consent           — all consent records
    GET  /consent/{cid}     — single client consent
    POST /validate-purpose  — purpose validation
    GET  /training-log      — training_log.json entries
    GET  /transparency-log  — transparency_log.json entries
    GET  /transparency-stats— summary statistics
    GET  /health            — health check
"""

import asyncio
import copy
import io
import json
import os
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Thread pool for CPU-heavy work (torch.load / torch.save / aggregation)
_executor = ThreadPoolExecutor(max_workers=4)

# ── Resolve project paths ─────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SYSTEM_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_SYSTEM_DIR)
sys.path.insert(0, _SYSTEM_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from models.pneumonia_cnn import PneumoniaCNN
from datasets.pneumonia_mnist_loader import get_dataset_features

from system.privacy.consent_manager import ConsentManager
from system.privacy.purpose_validator import PurposeValidator
from system.privacy.transparency_logger import TransparencyLogger

# ── Singletons ────────────────────────────────────────────────
_LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(_LOGS_DIR, exist_ok=True)

consent_mgr = ConsentManager(os.path.join(_LOGS_DIR, "consent_records.json"))
purpose_val = PurposeValidator(log_path=os.path.join(_LOGS_DIR, "purpose_violations.json"))
transparency = TransparencyLogger(os.path.join(_LOGS_DIR, "transparency_log.json"))

_TRAINING_LOG_PATH = os.path.join(_LOGS_DIR, "training_log.json")


def _read_training_log() -> List[Dict[str, Any]]:
    if os.path.exists(_TRAINING_LOG_PATH):
        try:
            with open(_TRAINING_LOG_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _write_training_log(entries: List[Dict[str, Any]]):
    with open(_TRAINING_LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)


# ── Global training state ─────────────────────────────────────
_lock = threading.RLock()  # reentrant: _log() can be called while _lock is held

_state: Dict[str, Any] = {
    "global_model": None,
    "current_round": 0,
    "total_rounds": 5,
    "local_epochs": 2,
    "batch_size": 32,
    "learning_rate": 0.001,
    "expected_clients": 2,
    "status": "idle",
    "client_weights": {},
    "round_history": [],
    "started_at": None,
    "finished_at": None,
    "output_lines": [],
}


def _log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with _lock:
        _state["output_lines"].append(line)
        if len(_state["output_lines"]) > 500:
            _state["output_lines"] = _state["output_lines"][-500:]


def _init_global_model():
    model = PneumoniaCNN(num_classes=2)
    _state["global_model"] = copy.deepcopy(model.state_dict())
    return model


# ── FedAvg aggregation ────────────────────────────────────────
def _fedavg_aggregate(client_updates: Dict[int, Dict]) -> OrderedDict:
    total_samples = sum(u["num_samples"] for u in client_updates.values())
    if total_samples == 0:
        return _state["global_model"]

    avg_state = OrderedDict()
    first_key = True
    for cid, update in client_updates.items():
        w = update["num_samples"] / total_samples
        sd = update["weights"]
        for key in sd:
            if first_key:
                avg_state[key] = sd[key].float() * w
            else:
                avg_state[key] += sd[key].float() * w
        first_key = False

    return avg_state


# ── Pydantic schemas ──────────────────────────────────────────
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


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="PFLlib Federated Aggregation Server", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    allow_credentials=True,
)


@app.on_event("startup")
def startup():
    _init_global_model()
    _log("Aggregation server initialized with fresh PneumoniaCNN model")


# ══════════════════════════════════════════════════════════════
#  CONSENT ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.post("/consent")
def post_consent(req: ConsentRequest):
    """Store consent with client_id, consent bool, and training purpose."""
    result = purpose_val.validate(
        purpose=req.purpose,
        dataset_features=get_dataset_features(),
        dataset_name="PneumoniaMNIST",
    )

    record = consent_mgr.grant_consent(req.client_id, req.consent)

    # Also store purpose alongside the consent record
    records = consent_mgr.all_records()
    key = str(req.client_id)
    if key in records:
        records[key]["purpose"] = req.purpose
        records[key]["purpose_valid"] = result["valid"]
        consent_mgr._records = records
        consent_mgr._save()

    _log(f"[Consent] Client {req.client_id}: consent={'GRANTED' if req.consent else 'DENIED'}, "
         f"purpose={req.purpose}, purpose_valid={result['valid']}")

    return {
        "client_id": req.client_id,
        "consent": record["consent"],
        "purpose": req.purpose,
        "purpose_valid": result["valid"],
        "timestamp": record["timestamp"],
        "message": (
            f"Consent recorded for purpose '{req.purpose}'. "
            + ("You may participate in training." if req.consent and result["valid"]
               else "Training blocked." if not req.consent
               else f"Purpose '{req.purpose}' is not valid for PneumoniaMNIST.")
        ),
    }


@app.get("/consent/{client_id}")
def get_consent(client_id: int):
    record = consent_mgr.get_consent(client_id)
    if record is None:
        return {"client_id": client_id, "consent": None, "message": "No consent record found."}
    return record


@app.get("/consent")
def get_all_consent():
    return consent_mgr.all_records()


# ══════════════════════════════════════════════════════════════
#  PURPOSE VALIDATION
# ══════════════════════════════════════════════════════════════

@app.post("/validate-purpose")
def validate_purpose(req: PurposeValidationRequest):
    return purpose_val.validate(req.purpose, req.dataset_features, "PneumoniaMNIST")


@app.get("/purpose-violations")
def get_purpose_violations():
    path = os.path.join(_LOGS_DIR, "purpose_violations.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


# ══════════════════════════════════════════════════════════════
#  TRAINING CONTROL
# ══════════════════════════════════════════════════════════════

@app.post("/start-round")
def start_round(
    total_rounds: int = 5,
    local_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    expected_clients: int = 2,
):
    """Begin a new federated training session."""
    with _lock:
        if _state["status"] == "waiting":
            raise HTTPException(409, "A round is already in progress.")

        result = purpose_val.validate(
            purpose="image_classification",
            dataset_features=get_dataset_features(),
            dataset_name="PneumoniaMNIST",
        )
        if not result["valid"]:
            raise HTTPException(400, f"Purpose validation failed: {result['message']}")

        # Reset transparency log for new session
        transparency._entries = []
        transparency._save()

        _write_training_log([])

        _init_global_model()
        _state.update({
            "current_round": 1,
            "total_rounds": total_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "expected_clients": expected_clients,
            "status": "waiting",
            "client_weights": {},
            "round_history": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "output_lines": [],
        })

    _log(f"=== Federated Training Started ===")
    _log(f"Dataset: PneumoniaMNIST | Rounds: {total_rounds} | Clients: {expected_clients}")
    _log(f"Local epochs: {local_epochs} | Batch size: {batch_size} | LR: {learning_rate}")
    _log(f"[PurposeValidator] ✅ image_classification validated for PneumoniaMNIST")
    _log(f"--- Waiting for Round 1 client uploads ---")

    return {
        "status": "waiting",
        "round": 1,
        "total_rounds": total_rounds,
        "message": "Training session started. Clients should begin local training.",
    }


@app.get("/global-model")
async def get_global_model():
    model_state = _state["global_model"]
    if model_state is None:
        raise HTTPException(404, "No global model available yet.")
    # Offload serialization to thread pool so event loop stays free
    loop = asyncio.get_event_loop()
    def _serialize():
        buf = io.BytesIO()
        torch.save(model_state, buf)
        buf.seek(0)
        return buf
    buf = await loop.run_in_executor(_executor, _serialize)
    return StreamingResponse(buf, media_type="application/octet-stream")


@app.get("/round-config")
def get_round_config():
    # No lock — read-only snapshot, keeps endpoint responsive during aggregation
    return {
        "current_round": _state["current_round"],
        "total_rounds": _state["total_rounds"],
        "local_epochs": _state["local_epochs"],
        "batch_size": _state["batch_size"],
        "learning_rate": _state["learning_rate"],
        "status": _state["status"],
    }


@app.post("/upload-weights")
async def upload_weights(
    client_id: int = Form(...),
    num_samples: int = Form(...),
    local_loss: float = Form(...),
    local_accuracy: float = Form(...),
    weights_file: UploadFile = File(...),
):
    """Client uploads locally-trained model weights."""
    # ── Consent check ──────────────────────────────────────────
    if not consent_mgr.has_consent(client_id):
        _log(f"[ConsentManager] ❌ Client {client_id} BLOCKED – no consent")
        transparency.log_exclusion(
            client_id=client_id,
            round_number=_state["current_round"],
            reason="consent_not_granted",
        )
        raise HTTPException(403, f"Client {client_id}: consent not granted.")

    # ── Purpose check ──────────────────────────────────────────
    consent_record = consent_mgr.get_consent(client_id)
    client_purpose = consent_record.get("purpose", "image_classification") if consent_record else "image_classification"

    if client_purpose != "image_classification":
        _log(f"[PurposeValidator] ❌ Client {client_id} BLOCKED – purpose mismatch "
             f"(consented='{client_purpose}', required='image_classification')")
        transparency.log_exclusion(
            client_id=client_id,
            round_number=_state["current_round"],
            reason=f"purpose_mismatch: '{client_purpose}'",
        )
        raise HTTPException(403, f"Client {client_id}: purpose mismatch.")

    purpose_result = purpose_val.validate(
        purpose="image_classification",
        dataset_features=get_dataset_features(),
        dataset_name="PneumoniaMNIST",
    )
    if not purpose_result["valid"]:
        _log(f"[PurposeValidator] ❌ Client {client_id} BLOCKED – purpose validation failed")
        transparency.log_exclusion(
            client_id=client_id,
            round_number=_state["current_round"],
            reason="purpose_validation_failed",
        )
        raise HTTPException(403, f"Client {client_id}: purpose validation failed.")

    # ── Read weights (offload blocking torch.load to thread pool) ──
    data = await weights_file.read()
    loop = asyncio.get_event_loop()
    local_state_dict = await loop.run_in_executor(
        _executor,
        lambda: torch.load(io.BytesIO(data), map_location="cpu", weights_only=False),
    )

    with _lock:
        current_round = _state["current_round"]
        _state["client_weights"][client_id] = {
            "weights": local_state_dict,
            "num_samples": num_samples,
            "local_loss": local_loss,
            "local_accuracy": local_accuracy,
        }

        _log(f"[Round {current_round}] Client {client_id}: uploaded weights "
             f"(samples={num_samples}, loss={local_loss:.4f}, acc={local_accuracy:.4f})")

        received = len(_state["client_weights"])
        expected = _state["expected_clients"]

    if received >= expected:
        # Run aggregation in a background thread so this response returns
        # immediately and the server stays responsive to other requests.
        threading.Thread(target=_do_aggregation, daemon=True).start()

    return {"status": "accepted", "client_id": client_id, "round": current_round}


def _do_aggregation():
    """Perform FedAvg and advance to next round (or finish)."""
    with _lock:
        current_round = _state["current_round"]
        client_weights = _state["client_weights"]
        _state["status"] = "aggregating"

    _log(f"[Round {current_round}] All clients reported. Performing FedAvg aggregation...")

    total_samples = sum(u["num_samples"] for u in client_weights.values())
    participating_ids = []
    excluded_ids = []
    contribution_weights = {}

    for cid, update in client_weights.items():
        w = update["num_samples"] / total_samples if total_samples > 0 else 0
        contribution_weights[cid] = round(w, 6)
        transparency.log_participation(
            client_id=cid,
            round_number=current_round,
            purpose="image_classification",
            features=["image", "label"],
            contribution_weight=w,
            dataset="PneumoniaMNIST",
            algorithm="FedAvg",
        )
        participating_ids.append(cid)

    new_global = _fedavg_aggregate(client_weights)

    losses = [u["local_loss"] for u in client_weights.values()]
    accs = [u["local_accuracy"] for u in client_weights.values()]
    avg_loss = float(np.mean(losses))
    avg_acc = float(np.mean(accs))

    round_record = {
        "round": current_round,
        "participants": list(client_weights.keys()),
        "excluded": excluded_ids,
        "avg_loss": round(avg_loss, 4),
        "avg_accuracy": round(avg_acc, 4),
        "per_client": {
            str(cid): {
                "local_loss": round(u["local_loss"], 4),
                "local_accuracy": round(u["local_accuracy"], 4),
                "num_samples": u["num_samples"],
                "contribution_weight": contribution_weights.get(cid, 0),
            }
            for cid, u in client_weights.items()
        },
        "contribution_weights": {str(cid): contribution_weights.get(cid, 0) for cid in client_weights},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    transparency.log_round_summary(
        round_number=current_round,
        participating_ids=participating_ids,
        excluded_ids=excluded_ids,
        purpose="image_classification",
        dataset="PneumoniaMNIST",
        algorithm="FedAvg",
    )

    # Append to training_log.json
    log_entry = {
        "round_number": current_round,
        "participating_clients": participating_ids,
        "excluded_clients": excluded_ids,
        "local_loss": {str(cid): round(u["local_loss"], 4) for cid, u in client_weights.items()},
        "avg_loss": round(avg_loss, 4),
        "avg_accuracy": round(avg_acc, 4),
        "contribution_weights": {str(cid): contribution_weights.get(cid, 0) for cid in client_weights},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    training_log = _read_training_log()
    training_log.append(log_entry)
    _write_training_log(training_log)

    _log(f"[Round {current_round}] ✅ Aggregation complete: avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.4f}")

    with _lock:
        _state["global_model"] = new_global
        _state["round_history"].append(round_record)
        _state["client_weights"] = {}

        if current_round >= _state["total_rounds"]:
            _state["status"] = "completed"
            _state["finished_at"] = datetime.now(timezone.utc).isoformat()
            _log(f"=== Training Complete ({_state['total_rounds']} rounds) ===")
            _log(f"Final avg accuracy: {avg_acc:.4f}")
        else:
            _state["current_round"] = current_round + 1
            _state["status"] = "waiting"
            _log(f"--- Waiting for Round {current_round + 1} client uploads ---")


# ══════════════════════════════════════════════════════════════
#  STATUS & TRANSPARENCY ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/status")
def get_status():
    # No lock — read-only snapshot, keeps endpoint responsive
    return {
        "status": _state["status"],
        "current_round": _state["current_round"],
        "total_rounds": _state["total_rounds"],
        "expected_clients": _state["expected_clients"],
        "clients_received": len(_state["client_weights"]),
        "round_history": _state["round_history"],
        "started_at": _state["started_at"],
        "finished_at": _state["finished_at"],
        "running": _state["status"] in ("waiting", "aggregating"),
        "output_lines": _state["output_lines"][-100:],
        "total_lines": len(_state["output_lines"]),
        "config": {
            "algorithm": "FedAvg",
            "dataset": "PneumoniaMNIST",
            "model": "PneumoniaCNN",
            "global_rounds": _state["total_rounds"],
            "local_epochs": _state["local_epochs"],
            "batch_size": _state["batch_size"],
            "learning_rate": _state["learning_rate"],
            "num_clients": _state["expected_clients"],
            "training_purpose": "image_classification",
            "dataset_features": ["image", "label"],
        },
    }


@app.get("/training-log")
def get_training_log():
    return _read_training_log()


@app.get("/transparency-log")
def get_transparency_log():
    return transparency.get_all_entries()


@app.get("/transparency-stats")
def get_transparency_stats():
    return transparency.get_summary_stats()


@app.get("/user-info/{user_id}")
def get_user_info(user_id: int):
    entries = transparency.get_all_entries()
    participations = [e for e in entries if e.get("event") == "participation" and e.get("client_id") == user_id]
    exclusions = [e for e in entries if e.get("event") == "exclusion" and e.get("client_id") == user_id]
    if not participations and not exclusions:
        return {"user_id": user_id, "message": "No training records found for this user."}

    weights = [p["contribution_weight"] for p in participations]
    avg_weight = round(sum(weights) / len(weights), 6) if weights else 0.0

    return {
        "user_id": user_id,
        "data_used": participations[-1].get("features", []) if participations else [],
        "training_rounds": len(participations),
        "contribution_weight": avg_weight,
        "contribution_history": weights,
        "purpose": participations[-1].get("purpose", "") if participations else "",
        "last_training": participations[-1].get("timestamp") if participations else None,
        "dataset": participations[-1].get("dataset", "") if participations else "",
        "algorithm": participations[-1].get("algorithm", "") if participations else "",
        "exclusions": len(exclusions),
    }


# ══════════════════════════════════════════════════════════════
#  COMPAT ENDPOINTS (dashboard polling)
# ══════════════════════════════════════════════════════════════

@app.get("/training-status")
def get_training_status():
    # No lock — read-only snapshot
    return {
        "job_id": f"federated_{_state['started_at']}" if _state["started_at"] else None,
        "status": _state["status"],
        "running": _state["status"] in ("waiting", "aggregating"),
        "started_at": _state["started_at"],
        "finished_at": _state["finished_at"],
        "exit_code": 0 if _state["status"] == "completed" else None,
        "config": {
            "algorithm": "FedAvg",
            "dataset": "PneumoniaMNIST",
            "model": "PneumoniaCNN",
            "global_rounds": _state["total_rounds"],
            "local_epochs": _state["local_epochs"],
            "batch_size": _state["batch_size"],
            "learning_rate": _state["learning_rate"],
            "num_clients": _state["expected_clients"],
            "training_purpose": "image_classification",
            "dataset_features": ["image", "label"],
        },
        "output_lines": _state["output_lines"][-100:],
        "total_lines": len(_state["output_lines"]),
        "round_history": _state["round_history"],
    }


@app.post("/start-training")
def start_training_compat(req: StartTrainingRequest):
    """Compatibility endpoint – dashboard calls this to start federated training."""
    return start_round(
        total_rounds=req.global_rounds,
        local_epochs=req.local_epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        expected_clients=req.num_clients,
    )


@app.post("/stop-training")
def stop_training():
    with _lock:
        if _state["status"] == "idle":
            raise HTTPException(404, "No training session is active.")
        _state["status"] = "completed"
        _state["finished_at"] = datetime.now(timezone.utc).isoformat()
    _log("=== Training STOPPED by user ===")
    return {"status": "stopped", "message": "Training session stopped."}


@app.get("/health")
def health():
    return {"status": "healthy", "role": "aggregation_server"}
