"""
Tests for the logging configuration module.
Validates handler setup, log file creation, and log level behavior.
"""

import os
import logging
import pytest
from unittest.mock import patch

from log_config import setup_logging


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset root logger handlers before each test to avoid state leakage."""
    root = logging.getLogger()
    root.handlers.clear()
    yield
    root.handlers.clear()


def test_setup_logging_creates_log_directory(tmp_path):
    """setup_logging must create the log directory if it doesn't exist."""
    log_dir = str(tmp_path / "test_logs")
    setup_logging(log_dir=log_dir)
    assert os.path.isdir(log_dir)


def test_setup_logging_creates_log_file(tmp_path):
    """setup_logging must create a timestamped log file."""
    log_dir = str(tmp_path / "test_logs")
    setup_logging(log_dir=log_dir)
    log_files = os.listdir(log_dir)
    assert len(log_files) == 1
    assert log_files[0].startswith("run_")
    assert log_files[0].endswith(".log")


def test_setup_logging_configures_two_handlers(tmp_path):
    """Root logger must have exactly 2 handlers: file + console."""
    log_dir = str(tmp_path / "test_logs")
    setup_logging(log_dir=log_dir)
    root = logging.getLogger()
    assert len(root.handlers) == 2


def test_setup_logging_file_handler_captures_debug(tmp_path):
    """File handler must capture DEBUG-level messages."""
    log_dir = str(tmp_path / "test_logs")
    setup_logging(log_dir=log_dir)

    test_logger = logging.getLogger("test_debug_capture")
    test_logger.debug("debug-test-message-12345")

    log_files = os.listdir(log_dir)
    log_file = os.path.join(log_dir, log_files[0])
    with open(log_file) as f:
        contents = f.read()
    assert "debug-test-message-12345" in contents


def test_setup_logging_idempotent(tmp_path):
    """Calling setup_logging twice must not create duplicate handlers."""
    log_dir = str(tmp_path / "test_logs")
    setup_logging(log_dir=log_dir)
    setup_logging(log_dir=log_dir)
    root = logging.getLogger()
    assert len(root.handlers) == 2


def test_third_party_loggers_suppressed(tmp_path):
    """Noisy third-party loggers must be set to WARNING level."""
    log_dir = str(tmp_path / "test_logs")
    setup_logging(log_dir=log_dir)
    assert logging.getLogger("matplotlib").level == logging.WARNING
    assert logging.getLogger("PIL").level == logging.WARNING
