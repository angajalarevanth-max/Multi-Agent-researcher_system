"""
Central logging setup for the API process.

Use :func:`setup_logging` once at startup (see ``main.lifespan``) and :func:`get_logger(__name__)`
in modules for hierarchical log names.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

_configured = False


def setup_logging(name: str = "multi_agent_researcher", level: Optional[str] = None) -> None:
    """
    Idempotent root logger configuration: console + optional rotating file under ./logs/.
    """
    global _configured
    if _configured:
        return

    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if os.getenv("LOG_TO_FILE", "false").lower() in ("1", "true", "yes"):
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a standard library logger under the configured root (no extra handler attachment)."""
    return logging.getLogger(name)
