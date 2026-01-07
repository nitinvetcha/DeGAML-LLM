"""
Configuration Management for DeGAML-LLM

This module provides utilities for loading, merging, and validating
configuration files (YAML) and command-line arguments.

Example:
    ```python
    import argparse
    from degaml.utils.config import load_config, merge_configs
    
    # Load from YAML file
    config = load_config("configs/model_configs/qwen0.5b.yaml")
    
    # Merge with CL arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    args = parser.parse_args()
    
    final_config = merge_configs(config, vars(args))
    ```
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path where to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.
    
    Later configs override earlier ones for conflicting keys.
    Nested dictionaries are merged recursively.
    
    Args:
        *configs: Variable number of configuration dictionaries
        
    Returns:
        Merged configuration dictionary
        
    Example:
        ```python
        base_config = {"model": {"lr": 1e-5, "layers": 12}}
        override_config = {"model": {"lr": 1e-4}}
        
        merged = merge_configs(base_config, override_config)
        # Result: {"model": {"lr": 1e-4, "layers": 12}}
        ```
    """
    if not configs:
        return {}
    
    result = {}
    
    for config in configs:
        if config is None:
            continue
            
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = merge_configs(result[key], value)
            else:
                # Override value
                result[key] = value
    
    return result


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """Validate that config contains all required keys.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required key paths (e.g., ["model.name", "training.lr"])
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []
    
    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                missing_keys.append(key_path)
                break
            current = current[key]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    return True


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get value from nested dictionary using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to value (e.g., "model.training.lr")
        default: Default value if path doesn't exist
        
    Returns:
        Value at key path, or default if not found
        
    Example:
        ```python
        config = {"model": {"training": {"lr": 1e-5}}}
        lr = get_nested_value(config, "model.training.lr")  # Returns 1e-5
        bs = get_nested_value(config, "model.training.batch_size", 32)  # Returns 32
        ```
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current


def set_nested_value(config: Dict[str, Any], key_path: str, value: Any):
    """Set value in nested dictionary using dot notation.
    
    Args:
        config: Configuration dictionary to modify
        key_path: Dot-separated path to value (e.g., "model.training.lr")
        value: Value to set
        
    Example:
        ```python
        config = {}
        set_nested_value(config, "model.training.lr", 1e-5)
        # Result: {"model": {"training": {"lr": 1e-5}}}
        ```
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
