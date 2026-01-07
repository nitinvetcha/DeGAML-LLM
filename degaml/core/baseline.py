#!/usr/bin/env python3
"""
Baseline Parameter Generation and Inference Module for DeGAML-LLM

This module generates LoRA adapters using the parameter generator (generalization module),
prepares them for use with Large Language Models, and evaluates each adapter by running
inference on test datasets.

This implements the baseline (no adaptation) component of DeGAML-LLM, generating adapters
directly from the task prompt using the hyperconvolutional decoder trained on collected
LoRA checkpoints.

Usage:
    python -m degaml.core.baseline \\
        --eval_dataset ARC-c \\
        --test_dataset ARC-c \\
        --num_samples 25 \\
        --inference_num_samples 0
    
    Args:
        eval_dataset: Dataset name for evaluation/adapter generation
        test_dataset: Dataset name for testing
        num_samples: Number of samples for adaptation (ablation studies)
        inference_num_samples: Samples for inference (0 = full dataset)
"""

import gc
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import accelerate.utils
import torch
from fire import Fire
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

# DeGAML imports
from degaml.generator.dataset import Text2Qwen25LoRA_FullCondDataset as Dataset
from degaml.generator.model import HyperConvDecoderModel_FullCond as Model
from degaml.generator.tokenizer import Qwen2505LoRA_Tokenizer2D as Tokenizer
from degaml.utils.paths import PathConfig

# Global settings
SEED = 999
torch.set_float32_matmul_precision("high")
accelerate.utils.set_seed(SEED)
os.environ["NUM_PROCESSES"] = "1"

# Initialize path configuration
paths = PathConfig()


def replicate_to_min_samples(data: List, min_samples: int = 128) -> List:
    """Replicate data to reach minimum sample count for hyperconvolutional decoder.
    
    The hyperconvolutional decoder requires exactly 128 samples for conditioning.
    When num_samples < 128, this function replicates samples to reach 128.
    
    Args:
        data: List of data samples
        min_samples: Minimum number of samples required (default: 128)
        
    Returns:
        List with at least min_samples elements (duplicated if necessary)
    """
    if len(data) >= min_samples:
        return data
    
    # Calculate repetitions needed (works for single-digit values too)
    repetitions = math.ceil(min_samples / len(data))
    replicated = (data * repetitions)[:min_samples]
    print(f"Replicated {len(data)} samples {repetitions}x to get {len(replicated)} samples for conditioning")
    return replicated


def generate_adapters(
    model: Model,
    loader: DataLoader,
    dataset: Dataset,
    eval_dataset: str,
    test_dataset: str,
    real_length: int
) -> List[str]:
    """Generate LoRA adapters using the parameter generator.
    
    Args:
        model: Parameter generator model
        loader: DataLoader with task prompts
        dataset: Dataset instance for saving checkpoints
        eval_dataset: Evaluation dataset name
        test_dataset: Test dataset name
        real_length: Number of adapters to generate
        
    Returns:
        List of paths to generated adapter checkpoint files
    """
    print("==> Generating adapters...")
    model.eval()
    
    adapter_paths = []
    save_root = paths.generated_root / "common_sense_reasoning"
    save_root.mkdir(parents=True, exist_ok=True)
    
    # Iterate through the dataloader to generate adapters
    for idx, (tokens, cond_id, cond_mask, tag) in enumerate(loader):
        if idx >= real_length:
            break

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mask = ~torch.isnan(tokens)
            tokens = torch.nan_to_num_(tokens, nan=0.0)
            conditions = {
                "input_ids": cond_id.to(device="cuda"),
                "attention_mask": cond_mask.to(device="cuda"),
            }
            
            print(f"Generating adapter {idx+1}/{real_length}...")
            predict = model(
                source=None,
                mask=mask.to("cuda"),
                condition=conditions,
                target=None,
                generate=True,
            )

        # Save the generated adapter checkpoint
        save_path = dataset.save_checkpoint(
            save_path=str(save_root / f"{eval_dataset}T_on_{test_dataset}V"),
            tokens=predict[0],
            tag=tag,
            number=idx
        )
        adapter_paths.append(save_path)
        torch.cuda.empty_cache()

    print(f"\nSuccessfully generated {len(adapter_paths)} adapters.")
    return adapter_paths


def run_inference_and_evaluate(
    test_dataset: str,
    num_adapters: int,
    eval_dataset: str,
    test_dataset_v: str,
    base_model_path: str,
    num_samples: Optional[int] = None
):
    """Run inference and calculate accuracy for each generated adapter.
    
    Args:
        test_dataset: Dataset name for testing
        num_adapters: Number of adapters to evaluate
        eval_dataset: Evaluation dataset name (for naming)
        test_dataset_v: Test dataset variant name
        base_model_path: Path to base LLM model
        num_samples: Optional number of samples for inference (None = full dataset)
    """
    print(f"==> Starting inference and evaluation for {num_adapters} adapters...")
    
    results_dir = paths.results_root / "common_sense_reasoning" / f"{test_dataset}_basic_inference"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: Inference scripts should be in data_root/prepare or similar
    # This assumes the LLaMAFactory scripts are available
    
    accuracies = {}

    for adapter_idx in range(num_adapters):
        print(f"\n--- Processing Adapter {adapter_idx}/{num_adapters-1} ---")
        
        adapter_path = paths.test_ckpts_root / f"{eval_dataset}T_on_{test_dataset_v}V_{adapter_idx}"
        # Include num_samples in filename to prevent caching issues
        samples_tag = f"_samples{num_samples}" if num_samples else ""
        output_file = results_dir / f"{eval_dataset}T_on_{test_dataset_v}V_adapter_{adapter_idx}{samples_tag}.jsonl"

        if not adapter_path.exists():
            print(f"Warning: Adapter path not found, skipping: {adapter_path}")
            continue

        print(f"Running inference with adapter: {adapter_path}")
        
        try:
            # Step 1: VLLM Inference (assumes LLaMAFactory scripts are available)
            infer_args = [
                "--model_name_or_path", str(base_model_path),
                "--save_name", str(output_file),
                "--dataset", f"{test_dataset}_test",
                "--adapter_name_or_path", str(adapter_path),
                "--vllm_config", '{"gpu_memory_utilization":0.3}',
            ]
            # Add num_samples limit for ablation studies
            if num_samples:
                infer_args.extend(["--max_samples", str(num_samples)])
            
            # Note: This requires LLaMAFactory's vllm_infer.py script to be available
            # Users should set up their data_root/prepare/scripts/ directory accordingly
            inference_script = paths.data_root / "prepare" / "scripts" / "vllm_infer.py"
            subprocess.run(["python", str(inference_script)] + infer_args, check=True)
            print(f"Inference complete. Results saved to: {output_file}")

            # Step 2: Accuracy Calculation using degaml.core.accuracy
            print(f"Calculating accuracy for adapter {adapter_idx}...")
            from degaml.core.accuracy import compute_accuracy
            
            result = compute_accuracy(str(output_file), task_type='auto')
            accuracies[adapter_idx] = result['accuracy']
            print(f"✅ Accuracy for adapter {adapter_idx}: {result['accuracy']*100:.2f}%")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during processing for adapter {adapter_idx}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred with adapter {adapter_idx}: {e}")

    print("\n\n📊 ==> Evaluation Summary <==")
    if accuracies:
        for idx, acc in sorted(accuracies.items()):
            print(f"Adapter {idx}: Accuracy = {acc*100:.2f}%")
        avg_acc = sum(accuracies.values()) / len(accuracies)
        print(f"\nAverage Accuracy across all adapters: {avg_acc*100:.2f}%")
    else:
        print("No accuracies were calculated.")


def find_safetensors_files(directory: Path) -> List[Path]:
    """Find all .safetensors files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of paths to .safetensors files
    """
    safetensors_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".safetensors"):
                safetensors_files.append(Path(root) / file)
    return safetensors_files


def copy_files(src_dir: Path, dest_dir: Path):
    """Recursively copy files from source to destination.
    
    Args:
        src_dir: Source directory
        dest_dir: Destination directory
    """
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
    for item in src_dir.iterdir():
        s = src_dir / item.name
        d = dest_dir / item.name
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def process_adapter_path(adapter_dir: Path, config_path: Path):
    """Process generated adapters into the format required for testing.
    
    Args:
        adapter_dir: Directory containing generated adapter checkpoints
        config_path: Path to model config directory
    """
    ckpts = find_safetensors_files(adapter_dir)
    for ckpt in ckpts:
        dir_name = adapter_dir.name + "_" + ckpt.stem
        save_dir = paths.test_ckpts_root / dir_name
        
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            # Copy base model config files
            copy_files(config_path, save_dir)
            # Copy and rename adapter weights
            shutil.copy2(ckpt, save_dir / "adapter_model.safetensors")


def main(
    eval_dataset: str,
    test_dataset: str,
    num_samples: Optional[int] = None,
    inference_num_samples: Optional[int] = None,
    model_name: str = "Qwen2.5-0.5B-Instruct",
    dataset_tag: str = "ARC-c",
    max_text_length: int = 384
):
    """Main function to run the baseline adapter generation and evaluation pipeline.
    
    Args:
        eval_dataset: Dataset used for evaluation/training adapters
        test_dataset: Dataset used for testing
        num_samples: Optional number of samples to use for adaptation (for ablation studies)
        inference_num_samples: Optional number of samples for inference (0 = full dataset)
        model_name: Base model name (default: Qwen2.5-0.5B-Instruct)
        dataset_tag: Dataset tag for checkpoints (default: ARC-c)
        max_text_length: Maximum text length for conditioning (default: 384)
    """
    if num_samples:
        print(f"Ablation mode: Using only {num_samples} samples")
    
    # Configure paths
    base_model_path = paths.get_model_path(model_name)
    extractor_path = paths.get_model_path("all-MiniLM-L12-v2")
    checkpoint_path = paths.get_checkpoint_path(f"qwen0.5lora__{dataset_tag}4000.pth")
    config_path = paths.config_root / "Qwen0.5"
    dataset_root = paths.data_root / "common_sense_reasoning"
    cond_root = paths.data_root / "prepare" / "data"
    
    # Ensure directories exist
    paths.ensure_directories()
    
    # Model configuration
    config = {
        "token_size": (10, 130),
        "real_length": 1,  # Number of adapters to generate
        "num_texts": 128,
        "criterion_weight": torch.load(
            dataset_root / dataset_tag / "criterion_weight.pt",
            map_location="cpu",
            weights_only=True
        ),
        "extractor_type": "BERT",
        "text_tokenizer": AutoTokenizer.from_pretrained(str(extractor_path)),
        "extra_condition_module": AutoModel.from_pretrained(
            str(extractor_path),
            torch_dtype="auto"
        ),
        "max_text_length": max_text_length,
        "model_config": {
            "features": [
                (128, max_text_length, 384), (128, 200, 300), (128, 100, 256),
                (256, 50, 200), (512, 50, 200), (1024, 25, 200),
                (1024, 10, 200), (2048, 10, 200), (4296, 10, 130),
            ],
            "condition_dim": (128, max_text_length, 384),
            "kernel_size": 9,
        },
        "device": "cuda",
    }
    
    # --- 1. Model and Data Loading ---
    print("==> Building parameter generator model...")
    diction = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

    model = Model(
        config=config["model_config"],
        criterion_weight=config["criterion_weight"].view(1, -1, 1, 1),
        extractor_type=config["extractor_type"],
        extra_condition_module=config["extra_condition_module"],
    )
    tokenizer = Tokenizer(token_size=config["token_size"])
    model.load_state_dict(diction, strict=False)
    model.to(config["device"])

    # Load test data (with optional subset for ablation)
    test_data_file = cond_root / f"{test_dataset}_test.json"
    with open(test_data_file, 'r', encoding='utf-8') as f:
        test_data_full = json.load(f)
    
    if num_samples:
        test_data = test_data_full[:num_samples]
        print(f"Using {len(test_data)} samples out of {len(test_data_full)} available")
    else:
        test_data = test_data_full

    # Replicate samples for conditioning if needed (hyperconvolutional decoder requires 128)
    conditioning_data = replicate_to_min_samples(test_data)

    test_set = Dataset(
        checkpoint_folders=[str(dataset_root / dataset_tag)],
        tokenizer=tokenizer,
        expected_iteration=None,
        real_length=config["real_length"],
        texts=[conditioning_data],
        num_texts=config["num_texts"],
        text_tokenizer=config["text_tokenizer"],
        max_text_length=config["max_text_length"],
    )
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        num_workers=0,
        collate_fn=test_set.collate_fn_test,
        shuffle=False,
    )

    # --- 2. Generate Adapters ---
    print(f"\nGenerating adapters based on prompts from {test_dataset}")
    adapter_paths = generate_adapters(
        model, test_loader, test_set, eval_dataset, test_dataset, config["real_length"]
    )
    
    # Free up memory before inference
    del model, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    if not adapter_paths:
        print("No adapters were generated. Exiting.")
        return

    # --- 3. Process Adapters ---
    print("\nProcessing generated adapters for inference...")
    adapter_dir = paths.generated_root / "common_sense_reasoning" / f"{eval_dataset}T_on_{test_dataset}V"
    process_adapter_path(adapter_dir, config_path)
    print("Adapter processing complete.")

    # --- 4. Run Inference and Evaluate ---
    # Determine inference samples:
    # - If inference_num_samples is None: use num_samples (default behavior)
    # - If inference_num_samples is 0: use full dataset (None)
    # - Otherwise: use inference_num_samples value
    if inference_num_samples is None:
        infer_samples = num_samples  # Default: use same as adaptation
    elif inference_num_samples == 0:
        infer_samples = None  # Explicit 0 means full dataset
    else:
        infer_samples = inference_num_samples
    
    run_inference_and_evaluate(
        test_dataset,
        len(adapter_paths),
        eval_dataset,
        test_dataset,
        base_model_path,
        num_samples=infer_samples
    )
    
    print(f"\n🎉 BASELINE EVALUATION COMPLETED! 🎉")


if __name__ == "__main__":
    Fire(main)
