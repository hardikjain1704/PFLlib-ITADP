"""
Privacy compliance hook for the PFLlib training pipeline.

DEPRECATED: This module is kept for backward compatibility.
The real compliance logic is now in system/privacy/ modules and is
integrated directly into serverbase.py.

If you still import from here, these functions are no-ops.
"""

import logging

logger = logging.getLogger(__name__)


def filter_consented_clients(selected_clients: list) -> list:
    """Deprecated no-op. Consent filtering is now in Server.select_clients()."""
    logger.warning("compliance_hook.filter_consented_clients is deprecated. "
                   "Consent is enforced in serverbase.py via privacy.ConsentManager.")
    return selected_clients


def log_training_round(client_id: int, weight: float):
    """Deprecated no-op. Logging is now in Server.receive_models()."""
    logger.warning("compliance_hook.log_training_round is deprecated. "
                   "Logging is handled in serverbase.py via privacy.TransparencyLogger.")
