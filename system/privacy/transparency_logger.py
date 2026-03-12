"""
Data Transparency Logger – logs real training participation to a persistent JSON file.

Each entry captures: client_id, round_number, purpose, features used,
contribution_weight, timestamp, and whether the client was excluded.
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class TransparencyLogger:
    """Logs all training participation events to a JSON file."""

    def __init__(self, log_path: str = None):
        if log_path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_path = os.path.join(base, "logs", "transparency_log.json")
        self._path = log_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._entries: List[Dict[str, Any]] = self._load()

    # ── Persistence ────────────────────────────────────────────

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._entries, f, indent=2, default=str)

    # ── Logging API ────────────────────────────────────────────

    def log_participation(
        self,
        client_id: int,
        round_number: int,
        purpose: str,
        features: List[str],
        contribution_weight: float,
        dataset: str = "",
        algorithm: str = "",
    ):
        """Log a client's participation in a training round."""
        entry = {
            "event": "participation",
            "client_id": client_id,
            "round_number": round_number,
            "purpose": purpose,
            "features": features,
            "contribution_weight": round(contribution_weight, 6),
            "dataset": dataset,
            "algorithm": algorithm,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._entries.append(entry)
            self._save()

    def log_exclusion(
        self,
        client_id: int,
        round_number: int,
        reason: str,
    ):
        """Log that a client was excluded from a training round."""
        entry = {
            "event": "exclusion",
            "client_id": client_id,
            "round_number": round_number,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._entries.append(entry)
            self._save()
        print(f"[TransparencyLogger] Client {client_id} excluded in round {round_number}: {reason}")

    def log_round_summary(
        self,
        round_number: int,
        participating_ids: List[int],
        excluded_ids: List[int],
        purpose: str,
        dataset: str = "",
        algorithm: str = "",
    ):
        """Log a summary of a full round."""
        entry = {
            "event": "round_summary",
            "round_number": round_number,
            "participating_clients": participating_ids,
            "excluded_clients": excluded_ids,
            "num_participating": len(participating_ids),
            "num_excluded": len(excluded_ids),
            "purpose": purpose,
            "dataset": dataset,
            "algorithm": algorithm,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._entries.append(entry)
            self._save()

    # ── Query API ──────────────────────────────────────────────

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Return all log entries."""
        with self._lock:
            return list(self._entries)

    def get_client_entries(self, client_id: int) -> List[Dict[str, Any]]:
        """Return log entries for a specific client."""
        with self._lock:
            return [e for e in self._entries if e.get("client_id") == client_id]

    def get_round_entries(self, round_number: int) -> List[Dict[str, Any]]:
        """Return log entries for a specific round."""
        with self._lock:
            return [e for e in self._entries if e.get("round_number") == round_number]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics from the log."""
        with self._lock:
            participations = [e for e in self._entries if e["event"] == "participation"]
            exclusions = [e for e in self._entries if e["event"] == "exclusion"]
            summaries = [e for e in self._entries if e["event"] == "round_summary"]

            # Per-client stats
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
                "client_stats": client_stats,
            }

    def clear(self):
        """Clear all log entries (for testing)."""
        with self._lock:
            self._entries = []
            self._save()
