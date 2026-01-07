"""
Path Configuration Management for DeGAML-LLM

This module provides centralized path management to replace hardcoded paths
throughout the codebase. All paths can be configured via environment variables
or programmatically.

Environment Variables:
    DEGAML_DATA_ROOT: Root directory for datasets (default: ./data)
    DEGAML_OUTPUT_ROOT: Root directory for outputs (default: ./outputs)
    DEGAML_CHECKPOINT_ROOT: Root directory for checkpoints (default: ./checkpoints)
    DEGAML_MODEL_ROOT: Root directory for models (default: ./models)
    DEGAML_CONFIG_ROOT: Root directory for configs (default: ./configs)

Example:
    ```python
    from degaml.utils.paths import PathConfig
    
    # Use defaults or environment variables
    paths = PathConfig()
    
    # Or override programmatically
    paths = PathConfig(data_root="/custom/data/path")
    
    # Access paths
    dataset_path = paths.get_dataset_path("ARC-c", split="test")
    model_path = paths.get_model_path("Qwen2.5-0.5B-Instruct")
    ```
"""

import os
from pathlib import Path
from typing import Optional, Union


class PathConfig:
    """Centralized path configuration for DeGAML-LLM.
    
    This class manages all file system paths used throughout DeGAML-LLM,
    replacing hardcoded paths with configurable alternatives.
    
    Attributes:
        data_root: Root directory for datasets
        output_root: Root directory for outputs
        checkpoint_root: Root directory for model checkpoints
        model_root: Root directory for pretrained models
        config_root: Root directory for configuration files
    """
    
    def __init__(
        self,
        data_root: Optional[str] = None,
        output_root: Optional[str] = None,
        checkpoint_root: Optional[str] = None,
        model_root: Optional[str] = None,
        config_root: Optional[str] = None,
    ):
        """Initialize path configuration.
        
        Args:
            data_root: Override for data root directory
            output_root: Override for output root directory
            checkpoint_root: Override for checkpoint root directory
            model_root: Override for model root directory
            config_root: Override for config root directory
        """
        self.data_root = Path(data_root or os.getenv("DEGAML_DATA_ROOT", "./data"))
        self.output_root = Path(output_root or os.getenv("DEGAML_OUTPUT_ROOT", "./outputs"))
        self.checkpoint_root = Path(checkpoint_root or os.getenv("DEGAML_CHECKPOINT_ROOT", "./checkpoints"))
        self.model_root = Path(model_root or os.getenv("DEGAML_MODEL_ROOT", "./models"))
        self.config_root = Path(config_root or os.getenv("DEGAML_CONFIG_ROOT", "./configs"))
        
        # Derived paths
        self.test_ckpts_root = self.output_root / "test_ckpts"
        self.results_root = self.output_root / "results"
        self.generated_root = self.output_root / "generated"
        self.slot_vectors_root = self.output_root / "slot_vectors"
        self.logs_root = self.output_root / "logs"
    
    def ensure_directories(self):
        """Create all directories if they don't exist."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Path) and attr_name.endswith(('_root', '_dir')):
                attr.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_path(
        self, 
        dataset: str, 
        split: str = "test",
        task_type: str = "common_sense_reasoning"
    ) -> Path:
        """Get path to a specific dataset file.
        
        Args:
            dataset: Dataset name (e.g., "ARC-c")
            split: Data split (e.g., "test", "train")
            task_type: Task type (e.g., "common_sense_reasoning", "math")
            
        Returns:
            Path to dataset JSON file
        """
        return self.data_root / task_type / f"{dataset}_{split}.json"
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path to a pretrained model.
        
        Args:
            model_name: Model name (e.g., "Qwen2.5-0.5B-Instruct")
            
        Returns:
            Path to model directory
        """
        return self.model_root / model_name
    
    def get_checkpoint_path(self, checkpoint_name: str) -> Path:
        """Get path to a checkpoint file.
        
        Args:
            checkpoint_name: Checkpoint filename (e.g., "qwen0.5lora__ARC-c4000.pth")
            
        Returns:
            Path to checkpoint file
        """
        return self.checkpoint_root / checkpoint_name
    
    def get_adapter_path(
        self,
        eval_dataset: str,
        test_dataset: str,
        adapter_idx: int = 0,
        suffix: str = ""
    ) -> Path:
        """Get path to an adapter directory.
        
        Args:
            eval_dataset: Evaluation dataset name
            test_dataset: Test dataset name  
            adapter_idx: Adapter index
            suffix: Optional suffix (e.g., "_ttl", "_ts_mixed_lambda0.5")
            
        Returns:
            Path to adapter directory
        """
        adapter_name = f"{eval_dataset}T_on_{test_dataset}V_{adapter_idx}{suffix}"
        return self.test_ckpts_root / adapter_name
    
    def get_results_path(
        self,
        dataset: str,
        result_type: str = "basic_inference",
        task_type: str = "common_sense_reasoning"
    ) -> Path:
        """Get path to results directory.
        
        Args:
            dataset: Dataset name
            result_type: Type of results (e.g., "basic_inference", "ttl_inference")
            task_type: Task type
            
        Returns:
            Path to results directory
        """
        return self.results_root / task_type / f"{dataset}_{result_type}"
    
    def get_config_path(self, config_name: str, config_type: str = "model") -> Path:
        """Get path to a configuration file.
        
        Args:
            config_name: Configuration name
            config_type: Type of config ("model" or "dataset")
            
        Returns:
            Path to config file or directory
        """
        if config_type == "model":
            return self.config_root / "model_configs" / f"{config_name}.yaml"
        elif config_type == "dataset":
            return self.config_root / "dataset_configs" / f"{config_name}.yaml"
        else:
            return self.config_root / config_name
    
    def __repr__(self) -> str:
        """String representation of path configuration."""
        return (
            f"PathConfig(\n"
            f"  data_root={self.data_root},\n"
            f"  output_root={self.output_root},\n"
            f"  checkpoint_root={self.checkpoint_root},\n"
            f"  model_root={self.model_root},\n"
            f"  config_root={self.config_root}\n"
            f")"
        )


# Global default instance
DEFAULT_PATHS = PathConfig()
