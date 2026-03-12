"""
Consent Manager – Persistent consent storage for FL clients.

Stores consent records in a JSON file so they persist across training runs
and can be read by the backend API / dashboard.
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ConsentManager:
    """Manages consent records in a persistent JSON file."""

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            storage_path = os.path.join(base, "logs", "consent_records.json")
        self._path = storage_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._records: Dict[str, Dict[str, Any]] = self._load()

    # ── Persistence ────────────────────────────────────────────

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._records, f, indent=2, default=str)

    # ── Public API ─────────────────────────────────────────────

    def grant_consent(self, client_id: int, consent: bool) -> Dict[str, Any]:
        """Record or update consent for a client."""
        with self._lock:
            key = str(client_id)
            record = {
                "client_id": client_id,
                "consent": consent,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._records[key] = record
            self._save()
            print(f"[ConsentManager] Client {client_id}: consent={'GRANTED' if consent else 'DENIED'}")
            return record

    def get_consent(self, client_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve consent record for a client."""
        with self._lock:
            return self._records.get(str(client_id))

    def has_consent(self, client_id: int) -> bool:
        """Return True if the client has explicitly granted consent.

        Returns True (allow) if no record exists – fail-open for legacy clients.
        """
        with self._lock:
            rec = self._records.get(str(client_id))
            if rec is None:
                return True  # no record → legacy client, allow by default
            return rec["consent"]

    def filter_consented_clients(self, selected_clients: list) -> list:
        """Filter a list of client objects; keep only those with consent.

        Returns (allowed_clients, excluded_ids).
        """
        allowed = []
        excluded = []
        for client in selected_clients:
            if self.has_consent(client.id):
                allowed.append(client)
            else:
                excluded.append(client.id)
                print(f"[ConsentManager] Client {client.id} EXCLUDED from round (no consent).")
        return allowed, excluded

    def all_records(self) -> Dict[str, Dict[str, Any]]:
        """Return all consent records."""
        with self._lock:
            return dict(self._records)

    def consented_client_ids(self) -> List[int]:
        """Return IDs of all clients that have granted consent."""
        with self._lock:
            return [
                rec["client_id"]
                for rec in self._records.values()
                if rec["consent"]
            ]

    def denied_client_ids(self) -> List[int]:
        """Return IDs of clients that have explicitly denied consent."""
        with self._lock:
            return [
                rec["client_id"]
                for rec in self._records.values()
                if not rec["consent"]
            ]
