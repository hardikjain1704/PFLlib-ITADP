"""
Transparency Service – provides per-user training participation data.
"""

from typing import Dict, Any, Optional
from data_store import store


class TransparencyService:
    """Returns transparency information for the Data Transparency Dashboard."""

    def get_user_info(self, user_id: int) -> Dict[str, Any]:
        info = store.get_user_info(user_id)
        if info is None:
            return {
                "user_id": user_id,
                "message": "No training records found for this user.",
            }
        return info

    def record_round(
        self,
        user_id: int,
        contribution_weight: float,
        data_used: list,
        purpose: str,
    ):
        store.record_training_round(user_id, contribution_weight, data_used, purpose)


transparency_service = TransparencyService()
