"""
Configuration loading and validation utilities.
"""

import os
from pathlib import Path
from typing import Dict, Any

import yaml
from loguru import logger

from .types import ConfigDict

def load_config(config_path: str) -> ConfigDict:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {str(e)}")

    validate_config(config)
    return config

def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration structure and values.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = [
        "document_processing",
        "fda_checker",
        "pattern_analysis",
        "rule_generation",
        "logging",
        "api"
    ]

    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate document processing config
    doc_proc = config["document_processing"]
    if "supported_formats" not in doc_proc:
        raise ValueError("Missing supported_formats in document_processing config")
    if not isinstance(doc_proc["supported_formats"], list):
        raise ValueError("supported_formats must be a list")

    # Validate FDA checker config
    fda_checker = config["fda_checker"]
    if "api_endpoint" not in fda_checker:
        raise ValueError("Missing api_endpoint in fda_checker config")

    # Validate pattern analysis config
    pattern_analysis = config["pattern_analysis"]
    required_pattern_fields = ["min_pattern_occurrence", "confidence_threshold"]
    for field in required_pattern_fields:
        if field not in pattern_analysis:
            raise ValueError(f"Missing {field} in pattern_analysis config")

    # Validate logging config
    logging_config = config["logging"]
    if "level" not in logging_config:
        raise ValueError("Missing level in logging config")
    if "file" not in logging_config:
        raise ValueError("Missing file in logging config")

    # Create log directory if it doesn't exist
    log_file = Path(logging_config["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

def get_config_value(config: ConfigDict, *keys: str, default: Any = None) -> Any:
    """Safely get a configuration value using nested keys.

    Args:
        config: Configuration dictionary
        *keys: Nested keys to traverse
        default: Default value if key doesn't exist

    Returns:
        Configuration value or default
    """
    current = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current 