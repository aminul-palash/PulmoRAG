"""Logging configuration for PulmoRAG"""

import logging
import os
import sys
from pythonjsonlogger import jsonlogger


def _get_log_level() -> str:
    """Get log level from environment to avoid circular import"""
    return os.getenv("LOG_LEVEL", "INFO")


def _is_debug() -> bool:
    """Check debug mode from environment to avoid circular import"""
    return os.getenv("DEBUG", "False").lower() == "true"


def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup logger with both console and file handlers"""
    log_level = _get_log_level()
    is_debug = _is_debug()

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))

    # Formatter
    if is_debug:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = jsonlogger.JsonFormatter()

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Global logger
logger = setup_logger("pulmorag")
