from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MementoRuntime:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.embedder = None
        self.casebank = None
        self.error_fix_bank = None
        self.skeletonizer = None
        self.schema_pruner = None
        self.validator = None

    @classmethod
    def build(cls, config: Any) -> "MementoRuntime":
        runtime = cls(config)
        from memory_module.casebank import CaseBank
        from memory_module.error_fix_bank import ErrorFixBank
        from memory_module.skeletonizer import SqlSkeletonizer
        from memory_module.validator import StaticValidator

        runtime.casebank = CaseBank()
        runtime.error_fix_bank = ErrorFixBank()
        runtime.skeletonizer = SqlSkeletonizer()
        runtime.validator = StaticValidator()
        return runtime


_RUNTIME: Optional[MementoRuntime] = None


def _fingerprint_config(config: Any) -> str:
    parts = []
    for key in ("enable", "train_policy", "eval_policy", "persist_dir"):
        if hasattr(config, key):
            parts.append(f"{key}={getattr(config, key)!r}")
    return ",".join(parts) if parts else repr(config)


def get_memento_runtime(config: Any) -> MementoRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = MementoRuntime.build(config)
        logger.info("Initialized MementoRuntime (config=%s).", _fingerprint_config(config))
    else:
        existing_fp = _fingerprint_config(_RUNTIME.config)
        incoming_fp = _fingerprint_config(config)
        if existing_fp != incoming_fp:
            logger.warning(
                "MementoRuntime already initialized; ignoring new config (existing=%s, incoming=%s).",
                existing_fp,
                incoming_fp,
            )
    return _RUNTIME
