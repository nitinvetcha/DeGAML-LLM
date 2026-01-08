#!/usr/bin/env python3
"""
mega.py - SAMKE Research Pipeline Orchestrator

This script coordinates the execution of multiple ML research scripts
following a strict, reproducible experimental pipeline.

Execution Flow:
    Step 0: Run BL.py → AC.py → baseline_acc
    Step 1: Run HG.py → generations.txt
    Step 2: For each hypothesis, dispatch to appropriate module, evaluate, assign reward
    Step 3: Convert to PT format → run PT.py → new LoRA adapter
    Step 4: Feed adapter to HG.py → repeat

Author: SAMKE Team
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# =============================================================================
# SECTION 1: PATH CONSTANTS (DO NOT MODIFY)
# =============================================================================

# Module Scripts (in SAMKE directory)
SAMKE_ROOT = "/home/nitin/SAMKE"
BL_SCRIPT = os.path.join(SAMKE_ROOT, "BL.py")
HG_SCRIPT = os.path.join(SAMKE_ROOT, "HG.py")
AC_SCRIPT = os.path.join(SAMKE_ROOT, "AC.py")
AC_LS_SCRIPT = os.path.join(SAMKE_ROOT, "AC_LS.py")
TS_SCRIPT = os.path.join(SAMKE_ROOT, "TS.py")
TL_SCRIPT = os.path.join(SAMKE_ROOT, "TL.py")
LM_SCRIPT = os.path.join(SAMKE_ROOT, "LM.py")
LS_SCRIPT = os.path.join(SAMKE_ROOT, "LS.py")
PT_SCRIPT = os.path.join(SAMKE_ROOT, "PT.py")

# Output Directories
OUTPUT_ROOT = "/home/nitin/SocraticX/Outputs"
RESULTS_ROOT = os.path.join(OUTPUT_ROOT, "results/common_sense_reasoning")
MEGA_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "mega_pipeline")

# Default Configuration
DEFAULT_EVAL_DATASET = "ARC-c"
DEFAULT_TEST_DATASET = "ARC-c"
DEFAULT_MODEL_PATH = "/home/nitin/SocraticX/Prereqs/models/Qwen2.5-0.5B-Instruct"

# Hypothesis Family → Script Mapping
FAMILY_SCRIPT_MAP = {
    "TTT": TL_SCRIPT,    # Test-Time Training → TL.py
    "TTS": TS_SCRIPT,    # Test-Time Scaling → TS.py
    "LORA": LM_SCRIPT,   # LoRA Mixing → LM.py
    "Latent": LS_SCRIPT, # Latent/SLOT → LS.py
}

# Hypothesis Family → Accuracy Script Mapping
FAMILY_AC_SCRIPT_MAP = {
    "TTT": AC_SCRIPT,
    "TTS": AC_SCRIPT,
    "LORA": AC_SCRIPT,
    "Latent": AC_LS_SCRIPT,  # Latent uses special accuracy calculation
}

# GPU Worker Configuration for Parallel Execution
DEFAULT_GPU_GROUPS = ["0,1", "2,3"]  # Two groups of 2 GPUs each

# Checkpoint Storage Paths
TEST_CKPTS_ROOT = os.path.join(OUTPUT_ROOT, "test_ckpts")
SLOT_VECTORS_ROOT = os.path.join(OUTPUT_ROOT, "slot_vectors")

# Evolution Mode Constants
EVOLUTION_POLICY = "policy"  # Only policy adapter evolves
EVOLUTION_MODEL = "model"    # Only base model/adapter evolves
EVOLUTION_CO = "co"          # Both evolve (co-evolution)


@dataclass
class IterationResult:
    """
    Stores the result of one iteration for chaining to next iteration.
    
    Attributes:
        iteration: The iteration number (1-indexed)
        best_family: Name of the best performing family (TTT, TTS, LORA, Latent)
        best_config: The hyperparameters of the best hypothesis
        best_accuracy: Accuracy on num_samples (becomes next iteration's baseline)
        policy_adapter_path: Path to trained policy adapter (for HG.py)
        base_checkpoint: Family-specific checkpoint info for model evolution
        exclude_tts: If True, exclude TTS from next iteration (TTS was best)
    """
    iteration: int
    best_family: str = ""
    best_config: Dict = field(default_factory=dict)
    best_accuracy: float = 0.0
    policy_adapter_path: Optional[str] = None
    base_checkpoint: Dict = field(default_factory=dict)
    exclude_tts: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "iteration": self.iteration,
            "best_family": self.best_family,
            "best_config": self.best_config,
            "best_accuracy": self.best_accuracy,
            "policy_adapter_path": self.policy_adapter_path,
            "base_checkpoint": self.base_checkpoint,
            "exclude_tts": self.exclude_tts,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "IterationResult":
        """Create from dictionary (for loading from JSON)."""
        return cls(
            iteration=data["iteration"],
            best_family=data.get("best_family", ""),
            best_config=data.get("best_config", {}),
            best_accuracy=data.get("best_accuracy", 0.0),
            policy_adapter_path=data.get("policy_adapter_path"),
            base_checkpoint=data.get("base_checkpoint", {}),
            exclude_tts=data.get("exclude_tts", False),
        )

# =============================================================================
# SECTION 2: LOGGING UTILITIES
# =============================================================================

class Logger:
    """Simple logger for pipeline execution."""
    
    def __init__(self, log_dir: str, run_id: str):
        self.log_dir = log_dir
        self.run_id = run_id
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"mega_run_{run_id}.log")
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
    
    def info(self, message: str):
        self.log(message, "INFO")
    
    def warning(self, message: str):
        self.log(message, "WARNING")
    
    def error(self, message: str):
        self.log(message, "ERROR")
    
    def success(self, message: str):
        self.log(message, "SUCCESS")


# =============================================================================
# SECTION 3: UTILITY FUNCTIONS
# =============================================================================

def run_script(script_path: str, args: List[str], logger: Logger, 
               env_vars: Optional[Dict] = None, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """
    Run a Python script with given arguments.
    
    Args:
        script_path: Path to the Python script
        args: List of command-line arguments
        logger: Logger instance
        env_vars: Optional environment variables to set
        cwd: Optional working directory
    
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = ["python", script_path] + args
    logger.info(f"Running: {' '.join(cmd)}")
    
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd
        )
        
        if result.returncode != 0:
            logger.error(f"Script failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr[:1000]}")
        
        return result.returncode, result.stdout, result.stderr
    
    except Exception as e:
        logger.error(f"Failed to run script: {e}")
        return -1, "", str(e)


def parse_generations(filepath: str, logger: Logger) -> List[Dict]:
    """
    Parse generations.txt output from HG.py.
    
    Format:
        --- Generation 1 / 20 ---
        
        {"family": "TTT", ...}
        
        ================================================================================
    
    Returns:
        List of hypothesis dicts with keys: family, hyperparameters, raw_json
    """
    logger.info(f"Parsing generations from: {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"Generations file not found: {filepath}")
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    hypotheses = []
    
    # Split by separator
    sections = content.split("=" * 80)
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # Find JSON in the section
        json_match = re.search(r'\{[^{}]*\}', section, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                hypothesis = json.loads(json_str)
                hypotheses.append({
                    "parsed": hypothesis,
                    "raw_json": json_str,
                    "family": hypothesis.get("family", "UNKNOWN")
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
                continue
    
    logger.info(f"Parsed {len(hypotheses)} hypotheses")
    return hypotheses


def extract_accuracy_from_output(stdout: str, logger: Logger) -> Optional[float]:
    """
    Parse accuracy from AC.py stdout output.
    
    Expected format contains line like:
        Accuracy: 75.50%
    """
    # Try different patterns
    patterns = [
        r"Accuracy:\s*([\d.]+)%",
        r"accuracy.*?:\s*([\d.]+)",
        r"acc.*?:\s*([\d.]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match:
            try:
                acc_str = match.group(1)
                # Handle percentage vs decimal
                acc = float(acc_str)
                if acc > 1.0:  # It's a percentage
                    acc = acc / 100.0
                logger.info(f"Extracted accuracy: {acc:.4f}")
                return acc
            except ValueError:
                continue
    
    logger.warning("Could not extract accuracy from output")
    return None


def find_results_file(
    dataset: str, 
    script_type: str, 
    logger: Logger,
    num_samples: int = None,
    hyperparams: Dict = None
) -> Optional[str]:
    """
    Find the results JSONL file based on the script type, dataset, and parameters.
    
    For each script type, builds the exact expected filename including:
    - Hyperparameters (slot_steps, slot_lr, lambda_val, etc.)
    - Sample count (samplesN or samplesfull)
    
    Args:
        dataset: Dataset name (e.g., "ARC-c")
        script_type: Script identifier ("BL", "TL", "TS", "LM", "LS")
        logger: Logger instance
        num_samples: Number of samples used (None = full dataset)
        hyperparams: Dict of hyperparameters used (family-specific)
    
    Returns:
        Path to the specific results file, or None if not found.
    """
    # Map script type to expected result directory patterns
    result_patterns = {
        "BL": f"{RESULTS_ROOT}/{dataset}_basic_inference",
        "TL": f"{RESULTS_ROOT}/{dataset}_ttl_inference",
        "TS": f"/home/nitin/SocraticX/Outputs/results_sing_diff/common_sense_reasoning/{dataset}",
        "LM": f"{RESULTS_ROOT}/{dataset}_ts_mixed_inference",
        "LS": f"{RESULTS_ROOT}/{dataset}_slot_inference",
    }
    
    result_dir = result_patterns.get(script_type)
    if not result_dir or not os.path.exists(result_dir):
        logger.warning(f"Results directory not found: {result_dir}")
        return None
    
    hyperparams = hyperparams or {}
    
    # Build the expected filename based on script type
    if script_type == "BL":
        # BL.py: {dstag_T}T_on_{test_dataset}V_adapter_{adapter_idx}_samples{N}.jsonl
        samples_tag = f"_samples{num_samples}" if num_samples else ""
        filename = f"{dataset}T_on_{dataset}V_adapter_0{samples_tag}.jsonl"
        
    elif script_type == "TL":
        # TL.py: {adapter_name}_results_samples{N}.jsonl
        samples_tag = f"_samples{num_samples}" if num_samples else ""
        # TL uses adapter_name from path, looking for pattern
        pattern = f"*_results{samples_tag}.jsonl"
        jsonl_files = list(Path(result_dir).glob(pattern))
        if jsonl_files:
            # Get most recent matching file
            latest = max(jsonl_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found results file: {latest}")
            return str(latest)
        else:
            logger.warning(f"No matching files for pattern: {pattern}")
            return None
    
    elif script_type == "TS":
        # TS.py: {eval_dataset}T_on_{test_dataset}V_final_samples{N}.jsonl
        samples_suffix = f"_samples{num_samples if num_samples else 'full'}"
        filename = f"{dataset}T_on_{dataset}V_final{samples_suffix}.jsonl"
        
    elif script_type == "LM":
        # LM.py: {dstag_T}T_on_{test_dataset}V_adapter_{adapter_idx}_samples{N}.jsonl
        samples_tag = f"_samples{num_samples}" if num_samples else ""
        filename = f"{dataset}T_on_{dataset}V_adapter_0{samples_tag}.jsonl"
        
    elif script_type == "LS":
        # LS.py: {dstag_T}T_on_{test_dataset}V_adapter_{adapter_idx}_slot_steps{X}_lr{Y}_samples{N}.jsonl
        slot_steps = hyperparams.get("slot_steps", 5)
        slot_lr = hyperparams.get("slot_lr", 0.1)
        samples_tag = f"samples{num_samples}" if num_samples else "samplesfull"
        filename = f"{dataset}T_on_{dataset}V_adapter_0_slot_steps{slot_steps}_lr{slot_lr}_{samples_tag}.jsonl"
    else:
        logger.warning(f"Unknown script type: {script_type}")
        return None
    
    # Build full path and check
    result_file = os.path.join(result_dir, filename)
    
    if os.path.exists(result_file):
        logger.info(f"Found results file: {result_file}")
        return result_file
    else:
        logger.warning(f"Expected results file not found: {result_file}")
        # Fallback to most recent file for backwards compatibility
        logger.warning("Falling back to most recent file (may be incorrect)")
        jsonl_files = list(Path(result_dir).glob("*.jsonl"))
        if jsonl_files:
            latest = max(jsonl_files, key=lambda p: p.stat().st_mtime)
            logger.warning(f"Using fallback file: {latest}")
            return str(latest)
        return None


def store_best_checkpoint(
    best_family: str,
    best_config: Dict,
    eval_dataset: str,
    test_dataset: str,
    num_samples: int,
    iter_dir: str,
    logger: Logger
) -> Dict:
    """
    Store the best hypothesis checkpoint info for use in next iteration.
    
    The checkpoint type varies by family:
    - TTT: Path to TTL-trained adapter (_ttl suffix)
    - LORA: Path to TS-mixed adapter (_ts_mixed_lambda{val} suffix)
    - Latent: Path to SLOT vector .pt file + original adapter path
    - TTS: List of adapter paths for ensembling (multiple adapters)
    
    Args:
        best_family: The winning family name
        best_config: The hyperparameters of the winning hypothesis
        eval_dataset: Evaluation dataset name
        test_dataset: Test dataset name
        num_samples: Number of samples used for adaptation
        iter_dir: Iteration output directory (for saving checkpoint info)
        logger: Logger instance
    
    Returns:
        Dict with checkpoint type and path(s)
    """
    checkpoint = {}
    
    if best_family == "TTT":
        # TTL adapter is saved at {adapter_path}_ttl by TL.py
        adapter_path = os.path.join(
            TEST_CKPTS_ROOT, 
            f"{eval_dataset}T_on_{test_dataset}V_0_ttl"
        )
        checkpoint = {
            "type": "ttl_adapter",
            "path": adapter_path,
            "config": best_config
        }
        logger.info(f"TTT checkpoint: {adapter_path}")
    
    elif best_family == "LORA":
        # TS-mixed adapter is saved with lambda in path by LM.py
        lambda_val = best_config.get("lambda", 0.5)
        adapter_path = os.path.join(
            TEST_CKPTS_ROOT,
            f"{eval_dataset}T_on_{test_dataset}V_0_ts_mixed_lambda{lambda_val}"
        )
        checkpoint = {
            "type": "ts_mixed",
            "path": adapter_path,
            "lambda_val": lambda_val,
            "config": best_config
        }
        logger.info(f"LORA checkpoint (TS-mixed): {adapter_path}")
    
    elif best_family == "Latent":
        # SLOT vector saved by LS.py + the original adapter
        slot_steps = best_config.get("slot_steps", 5)
        slot_lr = best_config.get("slot_lr", 0.1)
        samples_tag = f"samples{num_samples}" if num_samples else "samplesfull"
        adapter_name = f"{eval_dataset}T_on_{test_dataset}V_0"
        
        slot_path = os.path.join(
            SLOT_VECTORS_ROOT,
            f"{adapter_name}_slot_steps{slot_steps}_lr{slot_lr}_{samples_tag}.pt"
        )
        adapter_path = os.path.join(TEST_CKPTS_ROOT, adapter_name)
        
        checkpoint = {
            "type": "slot",
            "slot_path": slot_path,
            "adapter_path": adapter_path,
            "slot_steps": slot_steps,
            "slot_lr": slot_lr,
            "config": best_config
        }
        logger.info(f"Latent checkpoint: SLOT={slot_path}, Adapter={adapter_path}")
    
    elif best_family == "TTS":
        # TTS generates multiple adapters for ensembling
        num_adapters = best_config.get("num_lora_adpt", 3)
        adapter_paths = [
            os.path.join(TEST_CKPTS_ROOT, f"{eval_dataset}T_on_{test_dataset}V_{i}")
            for i in range(num_adapters)
        ]
        checkpoint = {
            "type": "tts_ensemble",
            "adapter_paths": adapter_paths,
            "num_adapters": num_adapters,
            "method": best_config.get("method", "majority_vote"),
            "config": best_config
        }
        logger.info(f"TTS checkpoint: {num_adapters} adapters for ensembling")
    
    else:
        logger.warning(f"Unknown family '{best_family}', no checkpoint stored")
        checkpoint = {"type": "unknown", "family": best_family}
    
    # Save checkpoint info to iteration directory
    checkpoint_info_path = os.path.join(iter_dir, "best_checkpoint.json")
    with open(checkpoint_info_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint info saved to: {checkpoint_info_path}")
    
    return checkpoint


def build_checkpoint_args(checkpoint: Optional[Dict], logger: Optional[Logger] = None) -> List[str]:
    """
    Build CLI arguments for checkpoint chaining based on checkpoint type.
    
    Args:
        checkpoint: Checkpoint dict from previous iteration (or None)
        logger: Optional logger for messages
    
    Returns:
        List of CLI arguments to pass to adaptation scripts
    """
    if not checkpoint:
        return []
    
    args = []
    ckpt_type = checkpoint.get("type", "")
    
    if ckpt_type == "tts_ensemble":
        # TTS case - pass all adapter paths as JSON list
        adapter_paths = checkpoint.get("adapter_paths", [])
        if adapter_paths:
            args.extend(["--base_adapter_paths", json.dumps(adapter_paths)])
            if logger:
                logger.info(f"Checkpoint chaining: Using {len(adapter_paths)} TTS adapters")
    
    elif ckpt_type == "ttl_adapter":
        # TTL adapter case
        adapter_path = checkpoint.get("path")
        if adapter_path:
            args.extend(["--base_adapter_path", adapter_path])
            if logger:
                logger.info(f"Checkpoint chaining: Using TTL adapter {adapter_path}")
    
    elif ckpt_type == "ts_mixed":
        # TS-mixed adapter case
        adapter_path = checkpoint.get("path")
        if adapter_path:
            args.extend(["--base_adapter_path", adapter_path])
            if logger:
                logger.info(f"Checkpoint chaining: Using TS-mixed adapter {adapter_path}")
    
    elif ckpt_type == "slot":
        # SLOT case - use the adapter path (SLOT vector will be recomputed)
        adapter_path = checkpoint.get("adapter_path")
        if adapter_path:
            args.extend(["--base_adapter_path", adapter_path])
            if logger:
                logger.info(f"Checkpoint chaining: Using adapter with SLOT {adapter_path}")
    
    return args


def convert_to_pt_format(
    hypotheses_with_rewards: List[Dict],
    output_dir: str,
    hg_prompt: str,
    task_name: str = "arc_challenge_task",
    logger: Optional[Logger] = None
) -> Tuple[str, str]:
    """
    Convert collected hypotheses and rewards to PT.py input format.
    
    Args:
        hypotheses_with_rewards: List of dicts with keys:
            - raw_json: The JSON hypothesis string
            - reward: 1 if improved, 0 otherwise
        output_dir: Directory to save output files
        hg_prompt: The hypothesis generation prompt used
        task_name: Task identifier for the JSON structure
        logger: Logger instance
    
    Returns:
        Tuple of (prompts_and_actions_path, results_path)
    """
    if logger:
        logger.info(f"Converting {len(hypotheses_with_rewards)} hypotheses to PT format")
    
    prompts_and_actions = {task_name: {}}
    results = {}
    
    for idx, item in enumerate(hypotheses_with_rewards):
        str_idx = str(idx)
        prompts_and_actions[task_name][str_idx] = {
            "prompt": hg_prompt,
            "response": item["raw_json"]
        }
        results[str_idx] = {
            "task_id": {"0": task_name},
            "correct": {"0": item["reward"] == 1}
        }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    actions_path = os.path.join(output_dir, "prompts_and_actions.json")
    results_path = os.path.join(output_dir, "results.json")
    
    with open(actions_path, "w", encoding="utf-8") as f:
        json.dump(prompts_and_actions, f, indent=2)
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    if logger:
        logger.info(f"Saved PT format files to: {output_dir}")
    
    return actions_path, results_path


def get_hg_prompt() -> str:
    """Extract the hypothesis generation prompt from HG.py for PT format conversion."""
    # This is a simplified version - in production, could parse from HG.py
    return """You are an expert AI agent tasked with generating an optimal adaptation strategy for a Large Language Model to improve its performance on a new, unseen task. Your goal is to output a single, structured JSON object that specifies the most promising strategy.

Task Context:
The AI2 Reasoning Challenge (ARC-Challenge) dataset is a benchmark for commonsense scientific reasoning. It contains multiple-choice science questions (grades 3-9) that require logical inference, multi-step reasoning, and application of factual knowledge rather than surface-level recall. Each question has one correct answer and several distractors.
Models performing well on ARC-Challenge must:
(a) Integrate scientific facts from diverse domains (physics, biology, chemistry, earth science).
(b) Demonstrate commonsense causal reasoning (e.g., why certain physical effects occur).
(c) Handle distractor filtering, requiring evidence-based elimination of implausible answers.

Your Instruction:
Based on the task context and the detailed method descriptions below, generate a single JSON object representing the most effective adaptation strategy. You must choose one strategy family and format your output strictly as a JSON object, with no additional text, explanations, or markdown.

JSON Output Format & Method Descriptions:
Your output must be a single JSON object with a "family" key and other keys corresponding to that family.

1.  If you choose the Test-Time Training family, use this format:

    ```json
    {
      "family": "TTT",
      "ttl_steps": <integer>,
      "learning_rate": <float>,
      "batch_size": <integer>,
      "shuffle_data": <boolean>
    }
    ```

    * Method Description: This strategy updates the model's weights at inference time using only unlabeled test data. Specifically, it performs self-supervised enhancement by minimizing the input perplexity.

2.  If you choose the LoRA Modification family, use this format:

    ```json
    {
      "family": "LORA",
      "lambda": <float>
    }
    ```

    * Method Description: This strategy uses the two-subspace (TS) mixing approach. Instead of the standard LoRA update ($A_1B_1 + A_2B_2$), this method introduces extra cross-term interactions by computing $(A_1+A_2)(B_1+B_2)$ to mix the two LoRA subspaces. The `lambda` hyperparameter determines the mixing ratio.

3.  If you choose the Test-Time Scaling family, use this format:

    ```json
    {
      "family": "TTS",
      "num_lora_adpt": <integer>,
      "method": "<string>"
    }
    ```

    * Method Description: This strategy involves generating multiple LoRA adapters, running inference with all of them, and then aggregating the predictions to get a final answer, for instance, by selecting the most confident prediction or using a majority vote.
    * The "method" key should contain one of the two specific string values - `"max_confidence"`, `"majority_vote"`.

4.  If you choose the Latent Space Modification family, use this format:

    ```json
    {
      "family": "Latent",
      "slot_steps": <integer>,
      "slot_lr": <float>
    }
    ```

    * Method Description: This strategy utilizes Sample-specific Language Model Optimization at Test-time (SLOT). It optimizes a lightweight, sample-specific parameter vector (delta) added to the final hidden layer features. By minimizing the cross-entropy loss on the input prompt itself over `slot_steps` (typically 1-5) with a learning rate of `slot_lr`, it aligns the model's internal representation to the specific instruction before generating the response.

Now, please provide only the JSON object for the ARC-c dataset.
"""

# =============================================================================
# SECTION 4: PIPELINE STAGES
# =============================================================================

def step0_baseline(
    eval_dataset: str,
    test_dataset: str,
    logger: Logger,
    adapter_path: Optional[str] = None,
    num_samples: int = None
) -> Optional[float]:
    """
    Step 0: Run baseline generation and calculate accuracy.
    
    Returns:
        Baseline accuracy as float, or None on failure
    """
    logger.info("=" * 60)
    logger.info("STEP 0: BASELINE GENERATION")
    logger.info("=" * 60)
    
    # Run BL.py
    bl_args = ["--eval_dataset", eval_dataset, "--test_dataset", test_dataset]
    if num_samples:
        bl_args.extend(["--num_samples", str(num_samples)])
    ret, stdout, stderr = run_script(BL_SCRIPT, bl_args, logger)
    
    if ret != 0:
        logger.error("BL.py failed")
        return None
    
    # Find results file
    results_file = find_results_file(test_dataset, "BL", logger, num_samples=num_samples)
    if not results_file:
        logger.error("Could not find baseline results file")
        return None
    
    # Run AC.py to calculate accuracy
    ret, stdout, stderr = run_script(AC_SCRIPT, [results_file], logger)
    
    if ret != 0:
        logger.error("AC.py failed")
        return None
    
    # Extract accuracy
    baseline_acc = extract_accuracy_from_output(stdout, logger)
    if baseline_acc is not None:
        logger.success(f"Baseline accuracy: {baseline_acc:.4f}")
    
    return baseline_acc


def step1_hypothesis_generation(
    output_file: str,
    logger: Logger,
    num_generations: int = 20,
    adapter_path: Optional[str] = None
) -> List[Dict]:
    """
    Step 1: Generate hypotheses using HG.py.
    
    Returns:
        List of parsed hypothesis dicts
    """
    logger.info("=" * 60)
    logger.info("STEP 1: HYPOTHESIS GENERATION")
    logger.info("=" * 60)
    
    # Build HG.py arguments
    hg_args = [
        "--output_file", output_file,
        "--num_generations", str(num_generations),
    ]
    
    # Add adapter path if provided (for iteration)
    if adapter_path:
        hg_args.extend(["--lora_adapter_path", adapter_path])
        logger.info(f"Using LoRA adapter: {adapter_path}")
    
    ret, stdout, stderr = run_script(HG_SCRIPT, hg_args, logger)
    
    if ret != 0:
        logger.error("HG.py failed")
        return []
    
    # Parse the generated hypotheses
    hypotheses = parse_generations(output_file, logger)
    
    if hypotheses:
        logger.success(f"Generated {len(hypotheses)} hypotheses")
        # Log family distribution
        family_counts = {}
        for h in hypotheses:
            fam = h["family"]
            family_counts[fam] = family_counts.get(fam, 0) + 1
        logger.info(f"Family distribution: {family_counts}")
    
    return hypotheses


def evaluate_single_hypothesis(
    hypothesis: Dict,
    hypothesis_idx: int,
    total_count: int,
    baseline_acc: float,
    eval_dataset: str,
    test_dataset: str,
    gpu_devices: str,
    log_dir: str,
    num_samples: int = None,
    prev_checkpoint: Dict = None
) -> Dict:
    """
    Evaluate a single hypothesis on specified GPU devices.
    
    This function is designed to run in a subprocess for parallel execution.
    It handles all evaluation logic for one hypothesis and returns the result.
    
    Args:
        hypothesis: The hypothesis dict with 'parsed', 'raw_json', 'family' keys
        hypothesis_idx: Index of this hypothesis (for logging)
        total_count: Total number of hypotheses (for logging)
        baseline_acc: Baseline accuracy to compare against
        eval_dataset: Evaluation dataset name
        test_dataset: Test dataset name
        gpu_devices: GPU devices to use (e.g., "0,1" or "2,3")
        log_dir: Directory for logging
        num_samples: Optional number of samples for ablation study
        prev_checkpoint: Checkpoint from previous iteration for model chaining
    
    Returns:
        Dict: The hypothesis with added 'accuracy' and 'reward' fields
    """
    # Create a worker-specific logger
    worker_id = gpu_devices.replace(",", "_")
    
    # Simple logging function for subprocess
    def log_msg(message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [GPU:{gpu_devices}] [{level}] {message}"
        print(log_entry)
        # Also append to a worker-specific log file
        log_file = os.path.join(log_dir, f"worker_{worker_id}.log")
        with open(log_file, "a") as f:
            f.write(log_entry + "\n")
    
    log_msg(f"Evaluating hypothesis {hypothesis_idx + 1}/{total_count}")
    log_msg(f"Family: {hypothesis['family']}")
    log_msg(f"Config: {hypothesis['parsed']}")
    
    family = hypothesis["family"]
    
    # Get the appropriate script
    script = FAMILY_SCRIPT_MAP.get(family)
    ac_script = FAMILY_AC_SCRIPT_MAP.get(family, AC_SCRIPT)
    
    if not script:
        log_msg(f"Unknown family '{family}', skipping", "WARNING")
        hypothesis["accuracy"] = 0.0
        hypothesis["reward"] = 0
        hypothesis["gpu_group"] = gpu_devices
        return hypothesis
    
    # Determine script type for finding results
    script_type_map = {
        TL_SCRIPT: "TL",
        TS_SCRIPT: "TS",
        LM_SCRIPT: "LM",
        LS_SCRIPT: "LS",
    }
    script_type = script_type_map.get(script, "UNKNOWN")
    
    # Build arguments for the adaptation script
    script_args = ["--eval_dataset", eval_dataset, "--test_dataset", test_dataset]
    
    parsed = hypothesis["parsed"]
    if family == "TTT":
        script_args.extend([
            "--ttl_steps", str(parsed.get("ttl_steps", 5)),
            "--learning_rate", str(parsed.get("learning_rate", 1e-5)),
            "--batch_size", str(parsed.get("batch_size", 4)),
            "--shuffle_data", str(parsed.get("shuffle_data", True)),
        ])
    elif family == "TTS":
        script_args.extend([
            "--num_lora_adpt", str(parsed.get("num_lora_adpt", 3)),
            "--method", parsed.get("method", "majority_vote"),
        ])
    elif family == "LORA":
        script_args.extend([
            "--lambda_val", str(parsed.get("lambda", 0.5)),
        ])
    elif family == "Latent":
        script_args.extend([
            "--slot_steps", str(parsed.get("slot_steps", 5)),
            "--slot_lr", str(parsed.get("slot_lr", 0.1)),
        ])
    
    # Add num_samples for ablation studies
    if num_samples:
        script_args.extend(["--num_samples", str(num_samples)])
    
    # Add checkpoint args for model chaining (if prev_checkpoint provided)
    if prev_checkpoint:
        # Create a simple logger adapter for build_checkpoint_args
        class SimpleLogger:
            def info(self, msg): log_msg(msg, "INFO")
            def warning(self, msg): log_msg(msg, "WARNING")
        
        checkpoint_args = build_checkpoint_args(prev_checkpoint, SimpleLogger())
        if checkpoint_args:
            script_args.extend(checkpoint_args)
            log_msg(f"Checkpoint args: {checkpoint_args}")
    
    log_msg(f"Script args: {script_args}")
    
    # Run the adaptation script with specified GPU devices
    cmd = ["python", script] + script_args
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_devices
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            log_msg(f"Adaptation script failed: {result.stderr[:500]}", "ERROR")
            hypothesis["accuracy"] = 0.0
            hypothesis["reward"] = 0
            hypothesis["gpu_group"] = gpu_devices
            return hypothesis
            
    except Exception as e:
        log_msg(f"Failed to run script: {e}", "ERROR")
        hypothesis["accuracy"] = 0.0
        hypothesis["reward"] = 0
        hypothesis["gpu_group"] = gpu_devices
        return hypothesis
    
    # Find results file (create a temporary logger for this)
    class TempLogger:
        def info(self, msg): log_msg(msg, "INFO")
        def warning(self, msg): log_msg(msg, "WARNING")
        def error(self, msg): log_msg(msg, "ERROR")
    
    temp_logger = TempLogger()
    # Pass num_samples and hyperparams for accurate file matching
    results_file = find_results_file(
        test_dataset, script_type, temp_logger, 
        num_samples=num_samples,
        hyperparams=parsed
    )
    
    if not results_file:
        log_msg(f"Could not find results file for hypothesis {hypothesis_idx}", "ERROR")
        hypothesis["accuracy"] = 0.0
        hypothesis["reward"] = 0
        hypothesis["gpu_group"] = gpu_devices
        return hypothesis
    
    # Calculate accuracy using appropriate AC script
    try:
        ac_result = subprocess.run(
            ["python", ac_script, results_file],
            capture_output=True,
            text=True,
            env=env
        )
        
        if ac_result.returncode != 0:
            log_msg(f"Accuracy calculation failed", "ERROR")
            hypothesis["accuracy"] = 0.0
            hypothesis["reward"] = 0
            hypothesis["gpu_group"] = gpu_devices
            return hypothesis
            
    except Exception as e:
        log_msg(f"Failed to run AC script: {e}", "ERROR")
        hypothesis["accuracy"] = 0.0
        hypothesis["reward"] = 0
        hypothesis["gpu_group"] = gpu_devices
        return hypothesis
    
    # Extract accuracy from output
    accuracy = extract_accuracy_from_output(ac_result.stdout, temp_logger)
    if accuracy is None:
        accuracy = 0.0
    
    # Assign reward
    reward = 1 if accuracy >= baseline_acc else 0
    
    hypothesis["accuracy"] = accuracy
    hypothesis["reward"] = reward
    hypothesis["gpu_group"] = gpu_devices
    
    log_msg(f"Accuracy: {accuracy:.4f} | Baseline: {baseline_acc:.4f} | Reward: {reward}")
    
    return hypothesis


def step2_hypothesis_loop(
    hypotheses: List[Dict],
    baseline_acc: float,
    eval_dataset: str,
    test_dataset: str,
    logger: Logger,
    gpu_groups: List[str] = None,
    num_samples: int = None,
    prev_checkpoint: Dict = None
) -> List[Dict]:
    """
    Step 2: Evaluate hypotheses in parallel across multiple GPU groups.
    
    Hypotheses are split across GPU groups and evaluated concurrently.
    For example, with gpu_groups=["0,1", "2,3"], half the hypotheses run on 
    GPUs 0,1 and the other half on GPUs 2,3 simultaneously.
    
    Args:
        hypotheses: List of hypothesis dicts to evaluate
        baseline_acc: Baseline accuracy for reward calculation
        eval_dataset: Evaluation dataset name
        test_dataset: Test dataset name
        logger: Logger instance
        gpu_groups: List of GPU device strings (e.g., ["0,1", "2,3"])
        num_samples: Optional number of samples for ablation study
        prev_checkpoint: Checkpoint from previous iteration for model chaining
    
    Returns:
        List of hypotheses with added 'accuracy' and 'reward' fields
    """
    if gpu_groups is None:
        gpu_groups = DEFAULT_GPU_GROUPS
    
    num_workers = len(gpu_groups)
    
    logger.info("=" * 60)
    logger.info("STEP 2: PARALLEL HYPOTHESIS EVALUATION")
    logger.info("=" * 60)
    logger.info(f"GPU Groups: {gpu_groups}")
    logger.info(f"Parallel workers: {num_workers}")
    logger.info(f"Total hypotheses: {len(hypotheses)}")
    
    # Assign GPU groups to hypotheses in round-robin fashion
    hypothesis_tasks = []
    for idx, hypothesis in enumerate(hypotheses):
        gpu_group = gpu_groups[idx % num_workers]
        hypothesis_tasks.append((hypothesis, idx, gpu_group))
    
    # Log the distribution
    for gpu_group in gpu_groups:
        count = sum(1 for _, _, g in hypothesis_tasks if g == gpu_group)
        logger.info(f"  GPU {gpu_group}: {count} hypotheses")
    
    results = [None] * len(hypotheses)  # Pre-allocate to maintain order
    
    # Use ProcessPoolExecutor for parallel evaluation
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {}
        for hypothesis, idx, gpu_group in hypothesis_tasks:
            future = executor.submit(
                evaluate_single_hypothesis,
                hypothesis,
                idx,
                len(hypotheses),
                baseline_acc,
                eval_dataset,
                test_dataset,
                gpu_group,
                logger.log_dir,
                num_samples,
                prev_checkpoint
            )
            futures[future] = idx
        
        # Collect results as they complete
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
                logger.info(f"Completed hypothesis {idx + 1}/{len(hypotheses)} "
                           f"on GPU {result.get('gpu_group', 'unknown')} - "
                           f"Accuracy: {result.get('accuracy', 0):.4f}")
            except Exception as e:
                logger.error(f"Hypothesis {idx + 1} failed with exception: {e}")
                # Create a failed result
                failed_hyp = hypotheses[idx].copy()
                failed_hyp["accuracy"] = 0.0
                failed_hyp["reward"] = 0
                failed_hyp["error"] = str(e)
                results[idx] = failed_hyp
    
    # Summary
    logger.info("=" * 40)
    logger.info("HYPOTHESIS EVALUATION SUMMARY")
    total_reward = sum(h["reward"] for h in results if h)
    logger.info(f"Total hypotheses: {len(results)}")
    logger.info(f"Successful (reward=1): {total_reward}")
    logger.info(f"Failed (reward=0): {len(results) - total_reward}")
    
    # Log results by GPU group
    for gpu_group in gpu_groups:
        group_results = [r for r in results if r and r.get("gpu_group") == gpu_group]
        group_rewards = sum(r["reward"] for r in group_results)
        logger.info(f"  GPU {gpu_group}: {group_rewards}/{len(group_results)} successful")
    
    return results


def step2b_best_family_final_evaluation(
    evaluated_hypotheses: List[Dict],
    baseline_acc: float,
    eval_dataset: str,
    test_dataset: str,
    logger: Logger,
    gpu_devices: str = "0,1",
    num_samples: int = None,
    iter_dir: str = None,
    skip_full_eval: bool = False
) -> Optional[Dict]:
    """
    Step 2B: Find best performing family, pick best config, store checkpoint,
    and optionally evaluate on FULL dataset.
    
    Algorithm:
    1. Count rewards by family (only hypotheses with reward=1)
    2. Find family with most successful hypotheses
    3. Within that family, pick the config with highest accuracy
    4. Store the best checkpoint for next iteration
    5. If not skip_full_eval: Run adaptation with num_samples, evaluate on FULL dataset
    
    Args:
        evaluated_hypotheses: List of evaluated hypothesis dicts
        baseline_acc: Baseline accuracy for comparison
        eval_dataset: Evaluation dataset name
        test_dataset: Test dataset name
        logger: Logger instance
        gpu_devices: GPU devices string
        num_samples: Number of samples for adaptation
        iter_dir: Iteration output directory
        skip_full_eval: If True, skip full dataset evaluation (for non-final iterations)
    
    Returns:
        Dict with evaluation results including checkpoint info, or None on failure
    """
    logger.info("=" * 60)
    logger.info("STEP 2B: BEST FAMILY FINAL EVALUATION")
    logger.info("=" * 60)
    
    if not num_samples:
        logger.info("No num_samples specified, skipping best family final evaluation")
        return None
    
    # Step 1: Count rewards by family
    family_rewards = {}
    for hyp in evaluated_hypotheses:
        if hyp and hyp.get("reward") == 1:
            family = hyp.get("family", "UNKNOWN")
            family_rewards[family] = family_rewards.get(family, 0) + 1
    
    if not family_rewards:
        logger.warning("No successful hypotheses found, skipping final evaluation")
        return None
    
    logger.info("Family reward counts:")
    for family, count in sorted(family_rewards.items(), key=lambda x: -x[1]):
        logger.info(f"  {family}: {count} successful hypotheses")
    
    # Step 2: Find family with most rewards
    best_family = max(family_rewards.keys(), key=lambda f: family_rewards[f])
    logger.success(f"Best family: {best_family} with {family_rewards[best_family]} successful hypotheses")
    
    # Step 3: Within that family, find the config with highest accuracy
    family_hypotheses = [h for h in evaluated_hypotheses if h and h.get("family") == best_family]
    best_hypothesis = max(family_hypotheses, key=lambda h: h.get("accuracy", 0))
    
    logger.info(f"Best config in {best_family} family:")
    parsed = best_hypothesis.get("parsed", {})
    logger.info(f"  Config: {parsed}")
    logger.info(f"  Accuracy (on {num_samples} samples): {best_hypothesis.get('accuracy', 0):.4f}")
    
    # Step 4: Store the best checkpoint for next iteration
    base_checkpoint = {}
    if iter_dir:
        base_checkpoint = store_best_checkpoint(
            best_family=best_family,
            best_config=parsed,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            num_samples=num_samples,
            iter_dir=iter_dir,
            logger=logger
        )
    
    # Build base result (checkpoint info is always computed)
    final_result = {
        "best_family": best_family,
        "family_reward_counts": family_rewards,
        "best_hypothesis": best_hypothesis,
        "ablation_accuracy": best_hypothesis.get("accuracy", 0),
        "adaptation_samples": num_samples,
        "base_checkpoint": base_checkpoint,
        "exclude_tts": best_family == "TTS",  # If TTS won, exclude it from next iteration
    }
    
    # Step 5: Skip full evaluation if requested (for non-final iterations)
    if skip_full_eval:
        logger.info("Skipping full dataset evaluation (skip_full_eval=True)")
        final_result["full_dataset_accuracy"] = None
        final_result["inference_samples"] = "skipped"
        
        # Save results to file
        if iter_dir:
            final_eval_path = os.path.join(iter_dir, "final_evaluation.json")
            with open(final_eval_path, "w") as f:
                json.dump(final_result, f, indent=2, default=str)
            logger.info(f"Final evaluation saved to: {final_eval_path}")
        
        return final_result
    
    # Run full dataset evaluation
    logger.info(f"Running final evaluation: adaptation with {num_samples} samples, inference on FULL dataset")
    
    family = best_hypothesis.get("family")
    script = FAMILY_SCRIPT_MAP.get(family)
    ac_script = FAMILY_AC_SCRIPT_MAP.get(family, AC_SCRIPT)
    
    if not script:
        logger.error(f"Unknown family '{family}', cannot run final evaluation")
        return None
    
    # Build script arguments
    script_args = [
        "--eval_dataset", eval_dataset,
        "--test_dataset", test_dataset,
        "--num_samples", str(num_samples),  # Use num_samples for adaptation
        "--inference_num_samples", "0",     # 0 means full dataset for inference
    ]
    
    # Add family-specific hyperparameters
    if family == "TTT":
        script_args.extend([
            "--ttl_steps", str(parsed.get("ttl_steps", 5)),
            "--learning_rate", str(parsed.get("learning_rate", 1e-5)),
            "--batch_size", str(parsed.get("batch_size", 4)),
            "--shuffle_data", str(parsed.get("shuffle_data", True)),
        ])
    elif family == "TTS":
        script_args.extend([
            "--num_lora_adpt", str(parsed.get("num_lora_adpt", 3)),
            "--method", parsed.get("method", "majority_vote"),
        ])
    elif family == "LORA":
        script_args.extend([
            "--lambda_val", str(parsed.get("lambda", 0.5)),
        ])
    elif family == "Latent":
        script_args.extend([
            "--slot_steps", str(parsed.get("slot_steps", 5)),
            "--slot_lr", str(parsed.get("slot_lr", 0.1)),
        ])
    
    logger.info(f"Running {os.path.basename(script)} with args: {script_args}")
    
    # Run the adaptation script
    cmd = ["python", script] + script_args
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_devices
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"Script failed: {result.stderr[:1000]}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to run script: {e}")
        return None
    
    # Find results file and calculate accuracy
    script_type_map = {
        TL_SCRIPT: "TL",
        TS_SCRIPT: "TS",
        LM_SCRIPT: "LM",
        LS_SCRIPT: "LS",
    }
    script_type = script_type_map.get(script, "UNKNOWN")
    
    # For step2b, we're running with num_samples for adaptation but inference_num_samples=0 (full dataset)
    # So the results file should be for full dataset (None), not num_samples
    results_file = find_results_file(
        test_dataset, script_type, logger,
        num_samples=None,  # inference was on full dataset (inference_num_samples=0)
        hyperparams=parsed
    )
    
    if not results_file:
        logger.error("Could not find results file for final evaluation")
        return None
    
    # Calculate accuracy
    try:
        ac_result = subprocess.run(
            ["python", ac_script, results_file],
            capture_output=True,
            text=True,
            env=env
        )
        
        if ac_result.returncode != 0:
            logger.error("Accuracy calculation failed")
            return None
            
    except Exception as e:
        logger.error(f"Failed to run AC script: {e}")
        return None
    
    # Extract accuracy
    final_accuracy = extract_accuracy_from_output(ac_result.stdout, logger)
    
    if final_accuracy is None:
        logger.warning("Could not extract final accuracy")
        final_accuracy = 0.0
    
    # Update result with full evaluation
    final_result["full_dataset_accuracy"] = final_accuracy
    final_result["inference_samples"] = "full"
    
    logger.info("=" * 40)
    logger.info("BEST FAMILY FINAL EVALUATION RESULTS")
    logger.info("=" * 40)
    logger.success(f"Best Family: {best_family}")
    logger.info(f"Best Config: {parsed}")
    logger.info(f"Accuracy on {num_samples} samples: {best_hypothesis.get('accuracy', 0):.4f}")
    logger.success(f"Accuracy on FULL dataset: {final_accuracy:.4f}")
    logger.info(f"Baseline accuracy: {baseline_acc:.4f}")
    
    improvement = final_accuracy - baseline_acc
    if improvement > 0:
        logger.success(f"Improvement over baseline: +{improvement:.4f}")
    else:
        logger.warning(f"Change from baseline: {improvement:.4f}")
    
    # Save results to file
    if iter_dir:
        final_eval_path = os.path.join(iter_dir, "final_evaluation.json")
        with open(final_eval_path, "w") as f:
            json.dump(final_result, f, indent=2, default=str)
        logger.info(f"Final evaluation saved to: {final_eval_path}")
    
    return final_result


def step3_policy_training(
    hypotheses_with_rewards: List[Dict],
    output_dir: str,
    logger: Logger,
    model_path: str = DEFAULT_MODEL_PATH
) -> Optional[str]:
    """
    Step 3: Convert data to PT format and train policy.
    
    Returns:
        Path to the new LoRA adapter directory, or None on failure
    """
    logger.info("=" * 60)
    logger.info("STEP 3: POLICY TRAINING")
    logger.info("=" * 60)
    
    # Convert to PT format
    pt_data_dir = os.path.join(output_dir, "pt_data")
    hg_prompt = get_hg_prompt()
    
    actions_path, results_path = convert_to_pt_format(
        hypotheses_with_rewards,
        pt_data_dir,
        hg_prompt,
        logger=logger
    )
    
    # Check if we have any successful hypotheses to train on
    successful_count = sum(1 for h in hypotheses_with_rewards if h["reward"] == 1)
    if successful_count == 0:
        logger.warning("No successful hypotheses to train on!")
        logger.info("Skipping policy training this iteration")
        return None
    
    logger.info(f"Training on {successful_count} successful hypotheses")
    
    # Define output directory for the new adapter
    adapter_output_dir = os.path.join(output_dir, "trained_adapter")
    
    # Build PT.py arguments
    pt_args = [
        "--actions_file", actions_path,
        "--results_file", results_path,
        "--model_name_or_path", model_path,
        "--output_dir", adapter_output_dir,
    ]
    
    # Run PT.py
    ret, stdout, stderr = run_script(PT_SCRIPT, pt_args, logger)
    
    if ret != 0:
        logger.error("PT.py failed")
        return None
    
    # Verify adapter was created
    if os.path.exists(adapter_output_dir):
        logger.success(f"New LoRA adapter created: {adapter_output_dir}")
        return adapter_output_dir
    else:
        logger.error("Adapter output directory not found after training")
        return None


# =============================================================================
# SECTION 5: MAIN ORCHESTRATION
# =============================================================================

def run_iteration(
    iteration: int,
    eval_dataset: str,
    test_dataset: str,
    logger: Logger,
    output_dir: str,
    policy_adapter_path: Optional[str] = None,
    num_hypotheses: int = 20,
    gpu_groups: List[str] = None,
    num_samples: int = None,
    prev_iteration_result: Optional[IterationResult] = None,
    evolution_mode: str = EVOLUTION_CO,
    is_final_iteration: bool = False
) -> Optional[IterationResult]:
    """
    Run a single iteration of the pipeline with evolution mode support.
    
    Args:
        iteration: The iteration number (1-indexed)
        eval_dataset: Evaluation dataset name
        test_dataset: Test dataset name
        logger: Logger instance
        output_dir: Root output directory
        policy_adapter_path: Path to policy adapter for HG.py
        num_hypotheses: Number of hypotheses to generate
        gpu_groups: List of GPU device groups
        num_samples: Number of samples for adaptation/ablation
        prev_iteration_result: Result from previous iteration (for chaining)
        evolution_mode: One of 'policy', 'model', 'co'
        is_final_iteration: If True, run full dataset evaluation
    
    Returns:
        IterationResult with all iteration data, or None on failure
    """
    iter_dir = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info(f"STARTING ITERATION {iteration}")
    logger.info(f"Evolution Mode: {evolution_mode}")
    if prev_iteration_result:
        logger.info(f"Previous Best Family: {prev_iteration_result.best_family}")
        logger.info(f"Previous Best Accuracy: {prev_iteration_result.best_accuracy:.4f}")
    logger.info("=" * 70)
    
    # Determine baseline accuracy
    # For iteration > 1 with model evolution, use previous iteration's best accuracy as baseline
    if prev_iteration_result and prev_iteration_result.best_accuracy > 0:
        baseline_acc = prev_iteration_result.best_accuracy
        logger.info(f"Using previous iteration's best accuracy as baseline: {baseline_acc:.4f}")
        # Save this for reference
        with open(os.path.join(iter_dir, "baseline.json"), "w") as f:
            json.dump({
                "baseline_accuracy": baseline_acc, 
                "iteration": iteration,
                "source": "prev_iteration",
                "prev_iteration": prev_iteration_result.iteration
            }, f, indent=2)
    else:
        # First iteration: compute baseline from BL.py
        baseline_acc = step0_baseline(eval_dataset, test_dataset, logger, policy_adapter_path, num_samples)
        if baseline_acc is None:
            logger.error("Failed to compute baseline accuracy")
            return None
        # Save baseline for reference
        with open(os.path.join(iter_dir, "baseline.json"), "w") as f:
            json.dump({"baseline_accuracy": baseline_acc, "iteration": iteration, "source": "BL.py"}, f, indent=2)
    
    # Step 1: Hypothesis Generation
    generations_file = os.path.join(iter_dir, "generations.txt")
    hypotheses = step1_hypothesis_generation(
        generations_file,
        logger,
        num_generations=num_hypotheses,
        adapter_path=policy_adapter_path  # Pass policy adapter to HG.py
    )
    
    if not hypotheses:
        logger.error("No hypotheses generated")
        return None
    
    # Filter out TTS hypotheses if previous iteration's best was TTS
    if prev_iteration_result and prev_iteration_result.exclude_tts:
        original_count = len(hypotheses)
        hypotheses = [h for h in hypotheses if h.get("family") != "TTS"]
        if len(hypotheses) < original_count:
            logger.info(f"Excluded {original_count - len(hypotheses)} TTS hypotheses (TTS was best in prev iteration)")
        
        if not hypotheses:
            logger.error("No non-TTS hypotheses remaining after filtering")
            return None
    
    # Step 2: Parallel Hypothesis Evaluation
    # Pass checkpoint from previous iteration for model chaining
    prev_checkpoint = None
    if evolution_mode in [EVOLUTION_MODEL, EVOLUTION_CO] and prev_iteration_result:
        prev_checkpoint = prev_iteration_result.base_checkpoint
        if prev_checkpoint:
            logger.info(f"Model evolution: Using checkpoint from iteration {prev_iteration_result.iteration}")
            logger.info(f"Checkpoint type: {prev_checkpoint.get('type', 'unknown')}")
    
    evaluated_hypotheses = step2_hypothesis_loop(
        hypotheses,
        baseline_acc,
        eval_dataset,
        test_dataset,
        logger,
        gpu_groups=gpu_groups,
        num_samples=num_samples,
        prev_checkpoint=prev_checkpoint
    )
    
    # Save evaluated hypotheses
    with open(os.path.join(iter_dir, "evaluated_hypotheses.json"), "w") as f:
        json.dump(evaluated_hypotheses, f, indent=2)
    
    # Step 2B: Best Family Evaluation
    # Skip full evaluation unless this is the final iteration
    final_eval_result = None
    if num_samples:
        final_eval_result = step2b_best_family_final_evaluation(
            evaluated_hypotheses,
            baseline_acc,
            eval_dataset,
            test_dataset,
            logger,
            gpu_devices=gpu_groups[0] if gpu_groups else "0,1",
            num_samples=num_samples,
            iter_dir=iter_dir,
            skip_full_eval=not is_final_iteration  # Only run full eval on final iteration
        )
    
    # Step 3: Policy Training (only if evolution_mode includes policy)
    new_policy_adapter_path = None
    if evolution_mode in [EVOLUTION_POLICY, EVOLUTION_CO]:
        new_policy_adapter_path = step3_policy_training(
            evaluated_hypotheses,
            iter_dir,
            logger
        )
    else:
        logger.info("Skipping policy training (evolution_mode='model')")
    
    # Build IterationResult
    result = IterationResult(iteration=iteration)
    
    if final_eval_result:
        result.best_family = final_eval_result.get("best_family", "")
        result.best_config = final_eval_result.get("best_hypothesis", {}).get("parsed", {})
        result.best_accuracy = final_eval_result.get("ablation_accuracy", 0.0)
        result.base_checkpoint = final_eval_result.get("base_checkpoint", {})
        result.exclude_tts = final_eval_result.get("exclude_tts", False)
    
    result.policy_adapter_path = new_policy_adapter_path
    
    # Save iteration result
    result_path = os.path.join(iter_dir, "iteration_result.json")
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    logger.info("=" * 70)
    logger.info(f"ITERATION {iteration} COMPLETE")
    logger.info(f"Best Family: {result.best_family}")
    logger.info(f"Best Accuracy: {result.best_accuracy:.4f}")
    if result.policy_adapter_path:
        logger.success(f"New policy adapter: {result.policy_adapter_path}")
    if result.base_checkpoint:
        logger.info(f"Base checkpoint type: {result.base_checkpoint.get('type', 'none')}")
    logger.info("=" * 70)
    
    return result


def run_single_variant(
    args,
    evolution_mode: str,
    run_id: str,
    logger: Logger,
    output_dir: str
) -> Dict:
    """
    Run a single evolution variant for the specified number of iterations.
    
    Args:
        args: Parsed CLI arguments
        evolution_mode: One of 'policy', 'model', 'co'
        run_id: Run identifier
        logger: Logger instance  
        output_dir: Output directory for this variant
        
    Returns:
        Dict with variant results including final accuracy
    """
    logger.info("=" * 70)
    logger.info(f"RUNNING VARIANT: {evolution_mode.upper()}")
    logger.info("=" * 70)
    
    prev_result = None
    current_policy_adapter = args.initial_adapter
    
    for iteration in range(1, args.iterations + 1):
        is_final = (iteration == args.iterations)
        
        # Determine what to pass based on evolution mode
        if evolution_mode == EVOLUTION_POLICY:
            # Policy only: update policy adapter, don't use prev model checkpoint
            pass_policy_adapter = current_policy_adapter
            pass_prev_result = None  # Don't chain model checkpoints
        elif evolution_mode == EVOLUTION_MODEL:
            # Model only: don't update policy adapter, chain model checkpoints
            pass_policy_adapter = args.initial_adapter  # Keep initial
            pass_prev_result = prev_result
        else:  # EVOLUTION_CO
            # Co-evolution: update both
            pass_policy_adapter = current_policy_adapter
            pass_prev_result = prev_result
        
        try:
            result = run_iteration(
                iteration=iteration,
                eval_dataset=args.eval_dataset,
                test_dataset=args.test_dataset,
                logger=logger,
                output_dir=output_dir,
                policy_adapter_path=pass_policy_adapter,
                num_hypotheses=args.num_hypotheses,
                gpu_groups=args.gpu_groups,
                num_samples=args.num_samples,
                prev_iteration_result=pass_prev_result,
                evolution_mode=evolution_mode,
                is_final_iteration=is_final
            )
            
            if result is None:
                logger.error(f"Iteration {iteration} returned None")
                break
            
            # Update for next iteration based on mode
            if evolution_mode in [EVOLUTION_POLICY, EVOLUTION_CO]:
                if result.policy_adapter_path:
                    current_policy_adapter = result.policy_adapter_path
            
            prev_result = result
            
        except Exception as e:
            logger.error(f"Iteration {iteration} failed with exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            break
    
    # Build variant result
    variant_result = {
        "evolution_mode": evolution_mode,
        "iterations_completed": prev_result.iteration if prev_result else 0,
        "final_family": prev_result.best_family if prev_result else "",
        "final_accuracy": prev_result.best_accuracy if prev_result else 0.0,
        "final_policy_adapter": current_policy_adapter,
        "final_checkpoint": prev_result.base_checkpoint if prev_result else {},
    }
    
    # Save variant result
    variant_result_path = os.path.join(output_dir, "variant_result.json")
    with open(variant_result_path, "w") as f:
        json.dump(variant_result, f, indent=2, default=str)
    
    return variant_result


def generate_comparison_report(results: Dict[str, Dict], output_dir: str, logger: Logger):
    """
    Generate a comparison report for all three variants.
    
    Args:
        results: Dict mapping mode name to variant result
        output_dir: Output directory for the report
        logger: Logger instance
    """
    logger.info("=" * 70)
    logger.info("GENERATING COMPARISON REPORT")
    logger.info("=" * 70)
    
    report = {
        "comparison_summary": [],
        "best_variant": None,
        "best_accuracy": 0.0,
    }
    
    for mode, result in results.items():
        summary = {
            "mode": mode,
            "final_accuracy": result.get("final_accuracy", 0.0),
            "final_family": result.get("final_family", ""),
            "iterations_completed": result.get("iterations_completed", 0),
        }
        report["comparison_summary"].append(summary)
        
        if result.get("final_accuracy", 0) > report["best_accuracy"]:
            report["best_accuracy"] = result["final_accuracy"]
            report["best_variant"] = mode
    
    # Log results
    logger.info("=" * 40)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 40)
    logger.info(f"{'Mode':<15} {'Accuracy':<12} {'Best Family':<15}")
    logger.info("-" * 45)
    for summary in report["comparison_summary"]:
        logger.info(f"{summary['mode']:<15} {summary['final_accuracy']:.4f}       {summary['final_family']:<15}")
    logger.info("-" * 45)
    logger.success(f"BEST VARIANT: {report['best_variant']} with accuracy {report['best_accuracy']:.4f}")
    
    # Save report
    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Comparison report saved to: {report_path}")
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SAMKE Research Pipeline Orchestrator - Multi-Iteration Evolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=DEFAULT_EVAL_DATASET,
        help="Evaluation dataset name"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=DEFAULT_TEST_DATASET,
        help="Test dataset name"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of pipeline iterations"
    )
    parser.add_argument(
        "--num_hypotheses",
        type=int,
        default=20,
        help="Number of hypotheses to generate per iteration"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=MEGA_OUTPUT_ROOT,
        help="Output directory for all pipeline artifacts"
    )
    parser.add_argument(
        "--initial_adapter",
        type=str,
        default=None,
        help="Path to initial LoRA adapter (optional)"
    )
    parser.add_argument(
        "--gpu_groups",
        type=str,
        nargs="+",
        default=DEFAULT_GPU_GROUPS,
        help="GPU device groups for parallel execution (e.g., '0,1' '2,3')"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configuration and exit without running"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use for ablation studies (e.g., 5, 15, 25)"
    )
    parser.add_argument(
        "--evolution_mode",
        type=str,
        choices=[EVOLUTION_POLICY, EVOLUTION_MODEL, EVOLUTION_CO],
        default=EVOLUTION_CO,
        help="Evolution mode: 'policy' (only policy evolves), 'model' (only model evolves), 'co' (both evolve)"
    )
    parser.add_argument(
        "--compare_all",
        action="store_true",
        help="Run all three evolution variants sequentially and compare results"
    )
    
    args = parser.parse_args()
    
    # Create run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine output directory structure based on mode
    if args.compare_all:
        base_output_dir = os.path.join(args.output_dir, f"comparison_{run_id}")
    else:
        base_output_dir = os.path.join(args.output_dir, f"run_{args.evolution_mode}_{run_id}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Initialize logger
    logger = Logger(base_output_dir, run_id)
    
    logger.info("=" * 70)
    logger.info("SAMKE MEGA PIPELINE ORCHESTRATOR - v2.0 (Multi-Iteration Evolution)")
    logger.info("=" * 70)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Eval Dataset: {args.eval_dataset}")
    logger.info(f"Test Dataset: {args.test_dataset}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Hypotheses per iteration: {args.num_hypotheses}")
    logger.info(f"GPU Groups: {args.gpu_groups}")
    logger.info(f"Output Directory: {base_output_dir}")
    logger.info(f"Initial Adapter: {args.initial_adapter or 'None'}")
    if args.num_samples:
        logger.info(f"Ablation Mode: Using {args.num_samples} samples")
    if args.compare_all:
        logger.info("COMPARISON MODE: Running all three evolution variants")
    else:
        logger.info(f"Evolution Mode: {args.evolution_mode}")
    
    if args.dry_run:
        logger.info("DRY RUN - Exiting without execution")
        return
    
    # Save configuration
    config = vars(args).copy()
    config["run_id"] = run_id
    with open(os.path.join(base_output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    if args.compare_all:
        # Run all three variants and compare
        comparison_results = {}
        
        for mode in [EVOLUTION_POLICY, EVOLUTION_MODEL, EVOLUTION_CO]:
            variant_output_dir = os.path.join(base_output_dir, f"variant_{mode}")
            os.makedirs(variant_output_dir, exist_ok=True)
            
            logger.info(f"\n{'#' * 70}")
            logger.info(f"# STARTING VARIANT: {mode.upper()}")
            logger.info(f"{'#' * 70}\n")
            
            variant_result = run_single_variant(
                args=args,
                evolution_mode=mode,
                run_id=run_id,
                logger=logger,
                output_dir=variant_output_dir
            )
            
            comparison_results[mode] = variant_result
            
            logger.info(f"\n{'#' * 70}")
            logger.info(f"# COMPLETED VARIANT: {mode.upper()}")
            logger.info(f"# Final Accuracy: {variant_result.get('final_accuracy', 0):.4f}")
            logger.info(f"{'#' * 70}\n")
        
        # Generate comparison report
        generate_comparison_report(comparison_results, base_output_dir, logger)
        
    else:
        # Run single variant
        variant_result = run_single_variant(
            args=args,
            evolution_mode=args.evolution_mode,
            run_id=run_id,
            logger=logger,
            output_dir=base_output_dir
        )
    
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"All outputs saved to: {base_output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

