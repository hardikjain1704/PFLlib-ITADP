"""
File-backed data store for privacy compliance records.

Reads consent from logs/consent_records.json and transparency from
logs/transparency_log.json – the same files written by the training pipeline.
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Resolve project root (backend_api/ is one level below project root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_CONSENT_PATH = os.path.join(_PROJECT_ROOT, 'logs', 'consent_records.json')
_TRANSPARENCY_PATH = os.path.join(_PROJECT_ROOT, 'logs', 'transparency_log.json')
_VIOLATIONS_PATH = os.path.join(_PROJECT_ROOT, 'logs', 'purpose_violations.json')


def _ensure_dirs():
    os.makedirs(os.path.join(_PROJECT_ROOT, 'logs'), exist_ok=True)

_ensure_dirs()


class ComplianceDataStore:
    """Thread-safe file-backed store for all compliance data."""

    def __init__(self):
        self._lock = threading.Lock()

    # ────────────────────────────────────────────────────────────
    #  Consent
    # ────────────────────────────────────────────────────────────

    def _read_consent(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(_CONSENT_PATH):
            try:
                with open(_CONSENT_PATH, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _write_consent(self, records: Dict[str, Dict[str, Any]]):
        with open(_CONSENT_PATH, 'w') as f:
            json.dump(records, f, indent=2, default=str)

    def set_consent(self, user_id: int, consent: bool) -> Dict[str, Any]:
        with self._lock:
            records = self._read_consent()
            record = {
                "client_id": user_id,
                "consent": consent,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            records[str(user_id)] = record
            self._write_consent(records)
            return record

    def get_consent(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            records = self._read_consent()
            return records.get(str(user_id))

    def has_consent(self, user_id: int) -> bool:
        with self._lock:
            records = self._read_consent()
            rec = records.get(str(user_id))
            if rec is None:
                return True  # legacy client – allow
            return rec["consent"]

    def all_consent_records(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return self._read_consent()

    # ────────────────────────────────────────────────────────────
    #  Transparency log
    # ────────────────────────────────────────────────────────────

    def _read_transparency(self) -> List[Dict[str, Any]]:
        if os.path.exists(_TRANSPARENCY_PATH):
            try:
                with open(_TRANSPARENCY_PATH, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def record_training_round(
        self, user_id: int, contribution_weight: float, data_used: list, purpose: str
    ):
        """Append a participation entry (used when called via the API endpoint)."""
        with self._lock:
            entries = self._read_transparency()
            entries.append({
                "event": "participation",
                "client_id": user_id,
                "round_number": len([e for e in entries if e.get("event") == "round_summary"]) + 1,
                "purpose": purpose,
                "features": data_used,
                "contribution_weight": round(contribution_weight, 6),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            with open(_TRANSPARENCY_PATH, 'w') as f:
                json.dump(entries, f, indent=2, default=str)

    def get_user_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Build per-client summary from transparency log."""
        with self._lock:
            entries = self._read_transparency()

        participations = [
            e for e in entries
            if e.get("event") == "participation" and e.get("client_id") == user_id
        ]
        exclusions = [
            e for e in entries
            if e.get("event") == "exclusion" and e.get("client_id") == user_id
        ]

        if not participations and not exclusions:
            return None

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

    def get_all_transparency_entries(self) -> List[Dict[str, Any]]:
        """Return all raw transparency log entries."""
        with self._lock:
            return self._read_transparency()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics from the transparency log."""
        with self._lock:
            entries = self._read_transparency()

        participations = [e for e in entries if e.get("event") == "participation"]
        exclusions = [e for e in entries if e.get("event") == "exclusion"]
        summaries = [e for e in entries if e.get("event") == "round_summary"]

        client_stats: Dict[int, Dict[str, Any]] = {}
        for p in participations:
            cid = p["client_id"]
            if cid not in client_stats:
                client_stats[cid] = {
                    "client_id": cid,
                    "rounds_participated": 0,
                    "total_weight": 0.0,
                    "contribution_history": [],
                    "features": p.get("features", []),
                    "purpose": p.get("purpose", ""),
                    "last_training": None,
                }
            cs = client_stats[cid]
            cs["rounds_participated"] += 1
            cs["total_weight"] += p["contribution_weight"]
            cs["contribution_history"].append(p["contribution_weight"])
            cs["last_training"] = p["timestamp"]

        for cid in client_stats:
            cs = client_stats[cid]
            n = cs["rounds_participated"]
            cs["avg_weight"] = round(cs["total_weight"] / n, 6) if n > 0 else 0.0

        return {
            "total_rounds": len(summaries),
            "total_participations": len(participations),
            "total_exclusions": len(exclusions),
            "unique_clients": len(client_stats),
            "client_stats": {str(k): v for k, v in client_stats.items()},
        }

    # ────────────────────────────────────────────────────────────
    #  Purpose violations
    # ────────────────────────────────────────────────────────────

    def get_violations(self) -> List[Dict[str, Any]]:
        if os.path.exists(_VIOLATIONS_PATH):
            try:
                with open(_VIOLATIONS_PATH, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def all_training_records(self) -> Dict[int, Dict[str, Any]]:
        """Backward compat: derive per-user training records from log."""
        stats = self.get_summary_stats()
        return stats.get("client_stats", {})


# Global singleton
store = ComplianceDataStore()
