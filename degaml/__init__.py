"""
DeGAML-LLM: Decoupled Generalization and Adaptation Meta-Learning for Large Language Models

This package implements DeGAML-LLM, a novel meta-learning framework that separates
generalization (parameter generation) from adaptation (task-specific refinement) for LLMs.

Main components:
- core: Main pipeline orchestration, hypothesis generation, baseline evaluation
- adaptation: Four adaptation family modules (TTT, TTS, LoRA, Latent)
- generator: Parameter generator (generalization module) from Drag-and-Drop-LLMs
- policy: RL policy for adaptation strategy selection
- utils: Shared utilities for configuration and path management
- ablation: Ablation study scripts

For usage examples, see the README.md and docs/ directory.
"""

__version__ = "1.0.0"
__author__ = "DeGAML-LLM Team"

from degaml.utils.paths import PathConfig
from degaml.utils.config import load_config

__all__ = ["PathConfig", "load_config"]
