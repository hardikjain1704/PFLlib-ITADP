"""
Purpose Limitation Validator – validates dataset features against the declared training purpose.

Raises a clear error and logs violations when a dataset contains features
not allowed for the specified purpose (e.g. 'phone_number' for image_classification).
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ── Default purpose rules ─────────────────────────────────────
DEFAULT_PURPOSE_RULES: Dict[str, List[str]] = {
    "image_classification": ["image", "label"],
    "text_classification": ["text", "label"],
    "sensor_classification": ["sensor", "label"],
}


class PurposeValidator:
    """Validates dataset features against purpose-limitation policies."""

    def __init__(
        self,
        purpose_rules: Optional[Dict[str, List[str]]] = None,
        log_path: Optional[str] = None,
    ):
        self.purpose_rules = purpose_rules or dict(DEFAULT_PURPOSE_RULES)
        if log_path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_path = os.path.join(base, "logs", "purpose_violations.json")
        self._log_path = log_path
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

    def validate(
        self,
        purpose: str,
        dataset_features: List[str],
        dataset_name: str = "",
    ) -> Dict[str, Any]:
        """Validate features; return result dict with 'valid' boolean.

        If invalid, also logs the violation and raises ValueError when
        called from the training pipeline (caller can choose to catch).
        """
        purpose_key = purpose.lower().replace(" ", "_")
        allowed = self.purpose_rules.get(purpose_key)

        if allowed is None:
            result = {
                "purpose": purpose,
                "valid": False,
                "message": f"Unknown purpose: '{purpose}'. Supported: {list(self.purpose_rules.keys())}",
                "allowed_features": [],
                "invalid_features": dataset_features,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._log_violation(result)
            return result

        invalid = [f for f in dataset_features if f not in allowed]
        valid = len(invalid) == 0

        result = {
            "purpose": purpose,
            "dataset": dataset_name,
            "dataset_features": dataset_features,
            "allowed_features": allowed,
            "invalid_features": invalid,
            "valid": valid,
            "message": (
                "Dataset validated. Training allowed."
                if valid
                else f"Training BLOCKED: features {invalid} violate purpose '{purpose}'."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not valid:
            self._log_violation(result)
            print(f"[PurposeValidator] ❌ VIOLATION: {result['message']}")
        else:
            print(f"[PurposeValidator] ✅ Dataset '{dataset_name}' passes purpose '{purpose}'.")

        return result

    def _log_violation(self, result: Dict[str, Any]):
        """Append violation to persistent log."""
        violations = []
        if os.path.exists(self._log_path):
            try:
                with open(self._log_path, "r") as f:
                    violations = json.load(f)
            except (json.JSONDecodeError, IOError):
                violations = []
        violations.append(result)
        with open(self._log_path, "w") as f:
            json.dump(violations, f, indent=2, default=str)

    def get_violations(self) -> List[Dict[str, Any]]:
        """Return all logged violations."""
        if os.path.exists(self._log_path):
            try:
                with open(self._log_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
