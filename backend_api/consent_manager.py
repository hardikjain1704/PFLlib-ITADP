"""
Consent Manager – handles user consent for federated learning participation.
"""

from typing import Dict, Any, Optional
from data_store import store


class ConsentManager:
    """Manages consent records for FL clients."""

    def grant_consent(self, user_id: int, consent: bool) -> Dict[str, Any]:
        record = store.set_consent(user_id, consent)
        return {
            "user_id": user_id,
            "consent": record["consent"],
            "timestamp": record["timestamp"],
            "message": (
                "Consent recorded. You may participate in training."
                if consent
                else "Training blocked. Consent required."
            ),
        }

    def get_consent(self, user_id: int) -> Dict[str, Any]:
        record = store.get_consent(user_id)
        if record is None:
            return {
                "user_id": user_id,
                "consent": None,
                "message": "No consent record found for this user.",
            }
        return {
            "user_id": user_id,
            "consent": record["consent"],
            "timestamp": record["timestamp"],
        }

    def check_training_allowed(self, user_id: int) -> bool:
        """Returns True if the client is allowed to train."""
        return store.has_consent(user_id)


consent_manager = ConsentManager()
