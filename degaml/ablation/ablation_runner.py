#!/usr/bin/env python3
"""
ablation_runner.py - Single-Family Ablation Study Runner

This script orchestrates 4 sequential runs of mega.py, each with a different
single-family filter (TTT, TTS, LORA, Latent). This allows for ablation study
where each run only provides one family option in the prompt.

Output Structure:
    mega_pipeline/ablation_run_{timestamp}/
        ├── TTT/
        │   └── iteration_1/...
        ├── TTS/
        │   └── iteration_1/...
        ├── Latent/
        │   └── iteration_1/...
        └── LORA/
            └── iteration_1/...

Usage:
    python ablation_runner.py --eval_dataset ARC-c --test_dataset ARC-c

Author: SAMKE Team
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to mega.py
MEGA_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mega.py")

# Output directory root
OUTPUT_ROOT = "/home/nitin/SocraticX/Outputs/mega_pipeline"

# Default ablation parameters
DEFAULT_NUM_SAMPLES = 25
DEFAULT_NUM_HYPOTHESES = 5
DEFAULT_ITERATIONS = 1

# The 4 families to run sequentially
FAMILIES = ["TTT", "TTS", "Latent", "LORA"]

# GPU groups for parallel hypothesis evaluation
DEFAULT_GPU_GROUPS = ["0,1", "2,3"]


# =============================================================================
# UTILITIES
# =============================================================================

class AblationLogger:
    """Simple logger for ablation runs."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "ablation_runner.log")
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
    
    def info(self, msg): self.log(msg, "INFO")
    def success(self, msg): self.log(msg, "SUCCESS")
    def warning(self, msg): self.log(msg, "WARNING")
    def error(self, msg): self.log(msg, "ERROR")


def run_mega_for_family(
    family: str,
    output_dir: str,
    eval_dataset: str,
    test_dataset: str,
    num_samples: int,
    num_hypotheses: int,
    iterations: int,
    gpu_groups: list,
    logger: AblationLogger
) -> bool:
    """
    Run mega.py for a single family.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 70)
    logger.info(f"STARTING SINGLE-FAMILY RUN: {family}")
    logger.info("=" * 70)
    
    # Build command
    cmd = [
        "python", MEGA_SCRIPT,
        "--eval_dataset", eval_dataset,
        "--test_dataset", test_dataset,
        "--iterations", str(iterations),
        "--num_hypotheses", str(num_hypotheses),
        "--num_samples", str(num_samples),
        "--output_dir", output_dir,
        "--family_filter", family,
        "--gpu_groups"
    ] + gpu_groups
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run mega.py and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream and log output
        family_log_file = os.path.join(output_dir, f"{family}_run.log")
        with open(family_log_file, "w") as f:
            for line in process.stdout:
                print(line, end="")  # Print to console
                f.write(line)  # Save to family-specific log
        
        process.wait()
        
        if process.returncode == 0:
            logger.success(f"✅ {family} run completed successfully")
            return True
        else:
            logger.error(f"❌ {family} run failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {family} run failed with exception: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for ablation runner."""
    parser = argparse.ArgumentParser(
        description="Single-Family Ablation Study Runner - Runs mega.py 4 times with different family filters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="ARC-c",
        help="Evaluation dataset name"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="ARC-c",
        help="Test dataset name"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of samples to use for ablation"
    )
    parser.add_argument(
        "--num_hypotheses",
        type=int,
        default=DEFAULT_NUM_HYPOTHESES,
        help="Number of hypotheses to generate per family"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of pipeline iterations per family"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_ROOT,
        help="Root output directory for ablation results"
    )
    parser.add_argument(
        "--gpu_groups",
        type=str,
        nargs="+",
        default=DEFAULT_GPU_GROUPS,
        help="GPU device groups for parallel execution"
    )
    parser.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=FAMILIES,
        choices=FAMILIES,
        help="Families to include in ablation (default: all 4)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configuration and exit without running"
    )
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_root = os.path.join(args.output_dir, f"ablation_run_{run_id}")
    os.makedirs(ablation_root, exist_ok=True)
    
    # Initialize logger
    logger = AblationLogger(ablation_root)
    
    logger.info("=" * 70)
    logger.info("SINGLE-FAMILY ABLATION STUDY RUNNER")
    logger.info("=" * 70)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Eval Dataset: {args.eval_dataset}")
    logger.info(f"Test Dataset: {args.test_dataset}")
    logger.info(f"Num Samples: {args.num_samples}")
    logger.info(f"Num Hypotheses: {args.num_hypotheses}")
    logger.info(f"Iterations per family: {args.iterations}")
    logger.info(f"GPU Groups: {args.gpu_groups}")
    logger.info(f"Families to run: {args.families}")
    logger.info(f"Output Directory: {ablation_root}")
    logger.info("=" * 70)
    
    # Save configuration
    config = {
        "run_id": run_id,
        "eval_dataset": args.eval_dataset,
        "test_dataset": args.test_dataset,
        "num_samples": args.num_samples,
        "num_hypotheses": args.num_hypotheses,
        "iterations": args.iterations,
        "gpu_groups": args.gpu_groups,
        "families": args.families,
        "output_dir": ablation_root,
    }
    
    with open(os.path.join(ablation_root, "ablation_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    if args.dry_run:
        logger.info("DRY RUN - Exiting without execution")
        return
    
    # Run each family sequentially
    results = {}
    
    for family in args.families:
        # Create family-specific output directory
        family_output_dir = os.path.join(ablation_root, family)
        os.makedirs(family_output_dir, exist_ok=True)
        
        # Run mega.py for this family
        success = run_mega_for_family(
            family=family,
            output_dir=family_output_dir,
            eval_dataset=args.eval_dataset,
            test_dataset=args.test_dataset,
            num_samples=args.num_samples,
            num_hypotheses=args.num_hypotheses,
            iterations=args.iterations,
            gpu_groups=args.gpu_groups,
            logger=logger
        )
        
        results[family] = "SUCCESS" if success else "FAILED"
        
        logger.info("-" * 70)
    
    # Summary
    logger.info("=" * 70)
    logger.info("ABLATION STUDY COMPLETE")
    logger.info("=" * 70)
    logger.info("Results Summary:")
    for family, status in results.items():
        icon = "✅" if status == "SUCCESS" else "❌"
        logger.info(f"  {icon} {family}: {status}")
    logger.info(f"All outputs saved to: {ablation_root}")
    logger.info("=" * 70)
    
    # Save results summary
    with open(os.path.join(ablation_root, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Exit with error code if any family failed
    if "FAILED" in results.values():
        sys.exit(1)


if __name__ == "__main__":
    main()
