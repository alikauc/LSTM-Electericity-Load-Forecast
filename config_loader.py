"""
Configuration loader for the LSTM Electricity Load Forecast project.

Loads settings from a YAML config file and provides CLI argument overrides.
Usage:
    cfg = load_config()            # loads config.yaml
    cfg = load_config("exp1.yaml") # loads custom config
"""

import argparse
import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def _deep_update(base: Dict, overrides: Dict) -> Dict:
    """
    Recursively merge overrides into base dict.
    Only updates keys that exist in base to prevent typos from silently creating new keys.
    """
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        elif key in base:
            base[key] = value
        else:
            logger.warning("Unknown config key ignored: '%s'", key)
    return base


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. Defaults to config.yaml in project root.

    Returns:
        Dictionary with all configuration values.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file contains invalid YAML.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty or contains invalid YAML: {config_path}")

    logger.info("Configuration loaded from: %s", config_path)
    return config


def parse_train_args() -> argparse.Namespace:
    """
    Parse CLI arguments for train.py, allowing override of any config value.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="LSTM Electricity Load Forecasting — Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file."
    )
    # Data overrides
    parser.add_argument("--csv-file", type=str, default=None, help="Override data.csv_path")
    # Training overrides
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override training.learning_rate")
    parser.add_argument("--patience", type=int, default=None, help="Override training.patience")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed")
    # Model overrides
    parser.add_argument("--hidden-size", type=int, default=None, help="Override model.hidden_size")
    parser.add_argument("--num-layers", type=int, default=None, help="Override model.num_layers")

    return parser.parse_args()


def get_train_config() -> Dict[str, Any]:
    """
    Load config from YAML and apply any CLI overrides.

    Returns:
        Final merged configuration dictionary.
    """
    args = parse_train_args()
    config = load_config(args.config)

    # Apply CLI overrides (only non-None values)
    cli_overrides = {
        "data": {},
        "training": {},
        "model": {},
    }

    if args.csv_file is not None:
        cli_overrides["data"]["csv_path"] = args.csv_file
    if args.epochs is not None:
        cli_overrides["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cli_overrides["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cli_overrides["training"]["learning_rate"] = args.lr
    if args.patience is not None:
        cli_overrides["training"]["patience"] = args.patience
    if args.seed is not None:
        cli_overrides["training"]["seed"] = args.seed
    if args.hidden_size is not None:
        cli_overrides["model"]["hidden_size"] = args.hidden_size
    if args.num_layers is not None:
        cli_overrides["model"]["num_layers"] = args.num_layers

    # Remove empty override groups
    cli_overrides = {k: v for k, v in cli_overrides.items() if v}

    if cli_overrides:
        _deep_update(config, cli_overrides)
        logger.info("CLI overrides applied: %s", cli_overrides)

    return config
