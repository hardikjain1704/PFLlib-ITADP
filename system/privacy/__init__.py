"""
Privacy compliance modules for PFLlib Federated Learning.

Three core components:
  - ConsentManager: Manages client consent with persistent JSON storage.
  - PurposeValidator: Validates dataset features against allowed purpose rules.
  - TransparencyLogger: Logs real training participation to JSON files.
"""

from .consent_manager import ConsentManager
from .purpose_validator import PurposeValidator
from .transparency_logger import TransparencyLogger

__all__ = ["ConsentManager", "PurposeValidator", "TransparencyLogger"]
