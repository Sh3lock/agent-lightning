from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MEMENTO_POLICY_SKELETON_ONLY = "skeleton_only"
MEMENTO_POLICY_TIERED = "tiered"
MEMENTO_VALID_POLICIES = {MEMENTO_POLICY_SKELETON_ONLY, MEMENTO_POLICY_TIERED}
DEFAULT_MEMENTO_TRAIN_POLICY = MEMENTO_POLICY_SKELETON_ONLY
DEFAULT_MEMENTO_EVAL_POLICY = MEMENTO_POLICY_TIERED


@dataclass(frozen=True)
class MementoConfig:
    enable: bool
    train_policy: str
    eval_policy: str


def _validate_memento_policy(value: str, env_key: str, default: str) -> str:
    if value in MEMENTO_VALID_POLICIES:
        return value
    logger.warning("Invalid %s=%s. Falling back to %s.", env_key, value, default)
    return default


def load_memento_config() -> MementoConfig:
    enable = os.environ.get("MEMENTO_ENABLE", "0") == "1"
    train_policy = os.environ.get("MEMENTO_TRAIN_POLICY", DEFAULT_MEMENTO_TRAIN_POLICY)
    eval_policy = os.environ.get("MEMENTO_EVAL_POLICY", DEFAULT_MEMENTO_EVAL_POLICY)
    train_policy = _validate_memento_policy(train_policy, "MEMENTO_TRAIN_POLICY", DEFAULT_MEMENTO_TRAIN_POLICY)
    eval_policy = _validate_memento_policy(eval_policy, "MEMENTO_EVAL_POLICY", DEFAULT_MEMENTO_EVAL_POLICY)
    return MementoConfig(enable=enable, train_policy=train_policy, eval_policy=eval_policy)
