"""
Centralized logging configuration for the LSTM Electricity Load Forecast project.

Provides a single setup_logging() function that configures both console and file
handlers. All modules should use:

    import logging
    logger = logging.getLogger(__name__)

Then call setup_logging() once at the start of main().
"""

import os
import logging
from datetime import datetime


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> None:
    """
    Configures the root logger with console and file handlers.

    Args:
        log_dir (str): Directory for log files. Created if it doesn't exist.
        log_level (int): Minimum level for file handler. Default is DEBUG.
        console_level (int): Minimum level for console handler. Default is INFO.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamped log filename for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if root_logger.handlers:
        root_logger.handlers.clear()

    # File handler — captures everything (DEBUG+) with full timestamps
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    # Console handler — shows INFO+ with a cleaner format
    console_formatter = logging.Formatter(
        fmt="%(levelname)-8s | %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("statsmodels").setLevel(logging.WARNING)

    logging.info(f"Logging initialized — file: {log_file}, console: {logging.getLevelName(console_level)}")
