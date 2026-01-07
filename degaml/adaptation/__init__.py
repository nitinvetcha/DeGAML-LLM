"""Adaptation family modules for DeGAML-LLM.

This package contains implementations of four adaptation families:
- test_time_training: Test-Time Training (TTT/TTL)
- test_time_scaling: Test-Time Scaling (TTS) via ensembling/routing
- lora_mixing: LoRA Subspace Mixing
- latent_space: Latent Space modification via SLOT vectors

Each module supports checkpoint chaining and integrates with the parameter generator.
"""

__all__ = [
    "test_time_training",
    "test_time_scaling",
    "lora_mixing",
    "latent_space",
]
