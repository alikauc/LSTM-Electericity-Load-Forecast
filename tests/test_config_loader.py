"""
Tests for the configuration loader module.
Validates YAML loading, CLI override merging, and error handling.
"""

import os
import pytest
import yaml

from config_loader import load_config, _deep_update


@pytest.fixture
def minimal_config(tmp_path):
    """Creates a minimal valid config YAML for testing."""
    config = {
        "data": {
            "csv_path": "test_data.csv",
            "csv_path_fallback": "fallback_data.csv",
            "features": ["Load_Calgary", "Temperature_C"],
            "train_ratio": 0.70,
            "val_ratio": 0.15,
        },
        "model": {
            "hidden_size": 32,
            "num_layers": 1,
        },
        "training": {
            "input_len": 24,
            "output_len": 24,
            "batch_size": 16,
            "epochs": 10,
            "patience": 3,
            "learning_rate": 0.001,
            "seed": 42,
        },
        "output": {
            "model_checkpoint": "models/test_model.pth",
        },
        "logging": {
            "log_dir": "logs",
            "log_level": "DEBUG",
            "console_level": "INFO",
        },
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


def test_load_config_returns_dict(minimal_config):
    """Config loader must return a dictionary."""
    cfg = load_config(minimal_config)
    assert isinstance(cfg, dict)


def test_load_config_has_required_sections(minimal_config):
    """Config must contain all required top-level sections."""
    cfg = load_config(minimal_config)
    for section in ["data", "model", "training", "output", "logging"]:
        assert section in cfg, f"Missing section: {section}"


def test_load_config_preserves_values(minimal_config):
    """Config values must match what was written to YAML."""
    cfg = load_config(minimal_config)
    assert cfg["training"]["epochs"] == 10
    assert cfg["training"]["learning_rate"] == 0.001
    assert cfg["model"]["hidden_size"] == 32
    assert cfg["data"]["features"] == ["Load_Calgary", "Temperature_C"]


def test_load_config_missing_file():
    """Loading a non-existent config file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_load_config_empty_file(tmp_path):
    """Loading an empty YAML file must raise ValueError."""
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("")
    with pytest.raises(ValueError, match="empty or contains invalid YAML"):
        load_config(str(empty_path))


def test_deep_update_merges_values():
    """_deep_update must recursively merge override values into base."""
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    overrides = {"a": {"x": 99}}
    result = _deep_update(base, overrides)
    assert result["a"]["x"] == 99
    assert result["a"]["y"] == 2  # Unchanged
    assert result["b"] == 3       # Unchanged


def test_deep_update_ignores_unknown_keys():
    """_deep_update must ignore keys not present in base dict."""
    base = {"a": 1}
    overrides = {"unknown_key": 99}
    result = _deep_update(base, overrides)
    assert "unknown_key" not in result
    assert result["a"] == 1


def test_load_default_config():
    """The default config.yaml in the project root must load successfully."""
    cfg = load_config()
    assert "training" in cfg
    assert cfg["training"]["seed"] == 42
