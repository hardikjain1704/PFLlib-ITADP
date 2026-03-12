"""
Purpose Limitation Validator – ensures datasets conform to the declared training purpose.
"""

from typing import Dict, Any, List

# ── Allowed feature sets per purpose ──────────────────────────
PURPOSE_RULES: Dict[str, List[str]] = {
    "image_classification": ["image", "label"],
    "text_classification": ["text", "label"],
}


class PurposeValidator:
    """Validates that a dataset's features are compatible with the declared purpose."""

    def validate(self, purpose: str, dataset_features: List[str]) -> Dict[str, Any]:
        purpose_key = purpose.lower().replace(" ", "_")
        allowed = PURPOSE_RULES.get(purpose_key)

        if allowed is None:
            return {
                "purpose": purpose,
                "valid": False,
                "message": f"Unknown purpose: '{purpose}'. Supported: {list(PURPOSE_RULES.keys())}",
                "allowed_features": [],
                "invalid_features": dataset_features,
            }

        invalid = [f for f in dataset_features if f not in allowed]
        valid = len(invalid) == 0

        return {
            "purpose": purpose,
            "dataset_features": dataset_features,
            "allowed_features": allowed,
            "invalid_features": invalid,
            "valid": valid,
            "message": (
                "Dataset validated. Training allowed."
                if valid
                else "Training blocked: dataset violates purpose limitation."
            ),
        }


purpose_validator = PurposeValidator()
