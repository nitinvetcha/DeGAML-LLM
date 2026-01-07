"""
P1 - Basic Parameter Generation and Inference Script with TS-Mixing (Corrected) (still Faulty)

This script generates a specified number of LoRA adapters, prepares them for use
with a Large Language Model (LLM), implements TS-Mixing LoRA, and then evaluates
each adapter individually by running inference on a test dataset and calculating the accuracy.
Finally, it plots a comparison of the accuracies.
This version is corrected to only process the adapters generated in the current run.
"""

import gc
import json
import os
import shutil
import subprocess
import sys

# --- Path and Environment Setup ---
# A file path is expected as a command-line argument, but we'll handle its absence
try:
    file_path = os.path.abspath(sys.argv[1])
except IndexError:
    print("Warning: No file path provided as a command-line argument. Using current directory as root.")
    file_path = os.path.abspath(__file__)

# Assuming 'SocraticX' is a key directory in your project structure
try:
    root_index = file_path.split(os.sep).index("SocraticX")
    root = os.sep.join(file_path.split(os.sep)[:root_index + 1])
except ValueError:
    # Fallback if 'SocraticX' is not in the path, adjust as needed
    root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(root)
# Add prerequisite path
sys.path.append('/home/nitin/SocraticX/Prereqs')
os.chdir(root)
os.environ["NUM_PROCESSES"] = "1"


# --- Constants and Configuration ---
TEST_ROOT = "/home/nitin/SocraticX/Outputs/test_ckpts"
RES_ROOT = "/home/nitin/SocraticX/Outputs/results/common_sense_reasoning"
DATASET_ROOT = "/home/nitin/SocraticX/Prereqs/data/common_sense_reasoning"
CONFIG_ROOT = "/home/nitin/SocraticX/Prereqs/datasets/common_sense_reasoning"
COND_ROOT = "/home/nitin/SocraticX/Prereqs/prepare/data"
SAVE_ROOT = "/home/nitin/SocraticX/Outputs/generated/common_sense_reasoning"
EXTRACTOR_PATH = "/home/nitin/SocraticX/Prereqs/models/all-MiniLM-L12-v2"
CONFIG_PATH = "/home/nitin/SocraticX/Prereqs/configs/Qwen0.5"
CHECKPOINT_PATH = "/home/nitin/SocraticX/Prereqs/checkpoints/qwen0.5lora__ARC-c4000.pth"
BASE_MODEL_PATH = "/home/nitin/SocraticX/Prereqs/models/Qwen2.5-0.5B-Instruct" # Relative to the 'prepare' dir

import torch
from fire import Fire
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from safetensors.torch import load_file, save_file


from degaml.generator.dataset import Text2Qwen25LoRA_FullCondDataset as Dataset
from degaml.generator.model import HyperConvDecoderModel_FullCond as Model
from degaml.generator.tokenizer import Qwen2505LoRA_Tokenizer2D as Tokenizer

# --- Global Settings ---
SEED = 999
torch.set_float32_matmul_precision("high")
import accelerate.utils
import math
accelerate.utils.set_seed(SEED)


def replicate_to_min_samples(data, min_samples=128):
    """Replicate data to reach minimum sample count for hyperconvolutional decoder.
    
    The hyperconvolutional decoder requires exactly 128 samples for conditioning.
    When num_samples < 128, this function replicates samples to reach 128.
    """
    if len(data) >= min_samples:
        return data
    
    # Calculate repetitions needed (works for single-digit values too)
    repetitions = math.ceil(min_samples / len(data))
    replicated = (data * repetitions)[:min_samples]
    print(f"Replicated {len(data)} samples {repetitions}x to get {len(replicated)} samples for conditioning")
    return replicated


max_text_length = 384
dataset_tag = "ARC-c"

config: dict = {
    "token_size": (10, 130),
    "real_length": 1, # Number of adapters to generate
    "num_texts": 128,
    "criterion_weight": torch.load(
        f"{CONFIG_ROOT}/{dataset_tag}/criterion_weight.pt", map_location="cpu", weights_only=True
    ),
    "extractor_type": "BERT",
    "text_tokenizer": AutoTokenizer.from_pretrained(EXTRACTOR_PATH),
    "extra_condition_module": AutoModel.from_pretrained(EXTRACTOR_PATH, torch_dtype="auto"),
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

# --- Core Functions ---

def generate_adapters(model, loader, dataset, dstag_T, dstag_V, real_length):
    """Generates a specified number of adapters."""
    print("==> Generating adapters...")
    model.eval()

    num_generated = 0
    # Iterate through the dataloader to generate the specified number of adapters
    for idx, (tokens, cond_id, cond_mask, tag) in enumerate(loader):
        if idx >= real_length:
            break

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mask = ~torch.isnan(tokens)
            tokens = torch.nan_to_num_(tokens, nan=0.0)
            conditions = {
                "input_ids": cond_id.to(device=config["device"]),
                "attention_mask": cond_mask.to(device=config["device"]),
            }

            print(f"Generating adapter {idx+1}/{real_length}...")
            predict = model(
                source=None,
                mask=mask.to(config["device"]),
                condition=conditions,
                target=None,
                generate=True,
            )

        # Save the generated adapter checkpoint
        dataset.save_checkpoint(
            save_path=f"{SAVE_ROOT}/{dstag_T}T_on_{dstag_V}V",
            tokens=predict[0],
            tag=tag,
            number=idx
        )
        num_generated += 1
        torch.cuda.empty_cache()

    print(f"\nSuccessfully generated {num_generated} adapters.")
    return num_generated

def run_inference_and_evaluate(test_dataset, num_adapters, dstag_T, dstag_V, adapter_type="basic", num_samples=None, lambda_val=0.5):
    """Runs inference and calculates accuracy for each generated adapter.
    
    Args:
        lambda_val: Lambda value for TS-mixing (used to locate correct adapter)
    """
    print(f"==> Starting inference and evaluation for {num_adapters} {adapter_type} adapters...")

    results_dir = f"{RES_ROOT}/{test_dataset}_{adapter_type}_inference"
    os.makedirs(results_dir, exist_ok=True)

    # Change directory to where inference scripts are located
    os.chdir("/home/nitin/SocraticX/Prereqs/prepare")

    accuracies = {}

    for adapter_idx in range(num_adapters):
        print(f"\n--- Processing Adapter {adapter_idx}/{num_adapters-1} ---")

        if adapter_type == "ts_mixed":
            # Include lambda_val in path to find the correct hyperparameter-specific adapter
            adapter_path = f"{TEST_ROOT}/{dstag_T}T_on_{test_dataset}V_{adapter_idx}_ts_mixed_lambda{lambda_val}"
        else:
            adapter_path = f"{TEST_ROOT}/{dstag_T}T_on_{test_dataset}V_{adapter_idx}"

        # Include num_samples in filename to prevent caching issues
        samples_tag = f"_samples{num_samples}" if num_samples else ""
        output_file = f"{results_dir}/{dstag_T}T_on_{test_dataset}V_adapter_{adapter_idx}{samples_tag}.jsonl"

        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter path not found, skipping: {adapter_path}")
            continue

        print(f"Running inference with adapter: {adapter_path}")

        try:
            # Step 1: VLLM Inference
            infer_args = [
                "--model_name_or_path", BASE_MODEL_PATH,
                "--save_name", output_file,
                "--dataset", f"{test_dataset}_test",
                "--adapter_name_or_path", adapter_path,
                "--vllm_config", '{"gpu_memory_utilization":0.4}',
            ]
            # Add num_samples limit for ablation studies
            if num_samples:
                infer_args.extend(["--max_samples", str(num_samples)])
            subprocess.run(["python", "scripts/vllm_infer.py"] + infer_args, check=True)
            print(f"Inference complete. Results saved to: {output_file}")

            # Step 2: Accuracy Calculation
            print(f"Calculating accuracy for adapter {adapter_idx}...")
            acc_args = ["--file", output_file]
            acc_result = subprocess.run(
                ["python", "scripts/calculate_acc.py"] + acc_args,
                check=True, capture_output=True, text=True
            )

            # Parse accuracy from the script's stdout
            accuracy_line = [line for line in acc_result.stdout.split('\n') if 'acc' in line]
            if accuracy_line:
                acc_val_str = accuracy_line[0].split(':')[-1].strip()
                accuracies[adapter_idx] = float(acc_val_str)
                print(f"✅ Accuracy for adapter {adapter_idx}: {acc_val_str}")
            else:
                print(f"⚠️ Warning: Could not parse accuracy for adapter {adapter_idx}.")
                print("STDOUT:", acc_result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"❌ Error during processing for adapter {adapter_idx}: {e}")
            if e.stdout: print("Stdout:", e.stdout)
            if e.stderr: print("Stderr:", e.stderr)
        except Exception as e:
            print(f"❌ An unexpected error occurred with adapter {adapter_idx}: {e}")

    print(f"\n\n📊 ==> {adapter_type.upper()} Evaluation Summary <==")
    if accuracies:
        for idx, acc in sorted(accuracies.items()):
            print(f"Adapter {idx}: Accuracy = {acc:.4f}")
        avg_acc = sum(accuracies.values()) / len(accuracies)
        print(f"\nAverage Accuracy across all {adapter_type} adapters: {avg_acc:.4f}")
    else:
        print(f"No accuracies were calculated for {adapter_type} adapters.")

    return accuracies

# --- New Functions for TS-Mixing and Plotting ---

def ts_mixing_lora(adapter_dir, num_generated, lambda_val=0.5):
    """Implements TS-Mixing for the specified number of newly generated LoRA adapters."""
    print("==> Implementing TS-Mixing LoRA...")

    # *** MODIFIED: Only process the newly generated adapters ***
    safetensors_files = [os.path.join(adapter_dir, f"{i}.safetensors") for i in range(num_generated)]

    for safetensor_file in safetensors_files:
        if not os.path.exists(safetensor_file):
            print(f"Warning: Safetensor file not found, skipping: {safetensor_file}")
            continue

        print(f"Processing file for TS-Mixing: {safetensor_file}")

        state_dict = load_file(safetensor_file)
        new_state_dict = state_dict.copy()

        for key, param in state_dict.items():
            if "lora_A" in key:
                # Decompose lora_A into two subspaces
                dim = param.size(0)
                if dim < 2: continue # Cannot split a dimension of size 1
                subspace1 = param[:dim//2, :]
                subspace2 = param[dim//2:, :]

                # Apply TS-Mixing formula
                mixed_subspace = lambda_val * subspace1 + (1 - lambda_val) * subspace2

                # Reconstruct the mixed lora_A
                mixed_param = param.clone()
                # Ensure correct slicing for odd dimensions
                mixed_param[:dim//2, :] = mixed_subspace
                mixed_param[dim//2:2*(dim//2), :] = mixed_subspace

                new_state_dict[key] = mixed_param

        # Save the new mixed adapter
        # Include lambda_val in directory name to avoid caching issues with different hyperparameters
        base_name = os.path.basename(safetensor_file).split('.')[0]
        dir_name = os.path.basename(adapter_dir) + "_" + base_name + f"_ts_mixed_lambda{lambda_val}"
        save_dir = os.path.join(TEST_ROOT, dir_name)

        # Always recreate the adapter (remove old if exists) to ensure correct lambda is applied
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        copy_files(CONFIG_PATH, save_dir)

        save_file(new_state_dict, os.path.join(save_dir, "adapter_model.safetensors"))
        print(f"Saved TS-Mixed adapter (lambda={lambda_val}) to: {save_dir}")


def ts_mixing_lora_single(adapter_path, lambda_val=0.5):
    """Applies TS-Mixing to a single existing adapter (for checkpoint chaining).
    
    Args:
        adapter_path: Path to the existing adapter directory
        lambda_val: Lambda value for mixing (0.0 to 1.0)
    """
    print(f"==> Applying TS-Mixing to existing adapter: {adapter_path}")
    
    # Find the adapter weights file
    safetensor_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(safetensor_file):
        print(f"Warning: adapter_model.safetensors not found in {adapter_path}")
        return None
    
    state_dict = load_file(safetensor_file)
    new_state_dict = state_dict.copy()
    
    for key, param in state_dict.items():
        if "lora_A" in key:
            # Decompose lora_A into two subspaces
            dim = param.size(0)
            if dim < 2: 
                continue  # Cannot split a dimension of size 1
            subspace1 = param[:dim//2, :]
            subspace2 = param[dim//2:, :]
            
            # Apply TS-Mixing formula
            mixed_subspace = lambda_val * subspace1 + (1 - lambda_val) * subspace2
            
            # Reconstruct the mixed lora_A
            mixed_param = param.clone()
            mixed_param[:dim//2, :] = mixed_subspace
            mixed_param[dim//2:2*(dim//2), :] = mixed_subspace
            
            new_state_dict[key] = mixed_param
    
    # Save the new mixed adapter
    adapter_name = os.path.basename(adapter_path)
    dir_name = f"{adapter_name}_ts_mixed_lambda{lambda_val}"
    save_dir = os.path.join(TEST_ROOT, dir_name)
    
    # Always recreate the adapter to ensure correct lambda is applied
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    copy_files(CONFIG_PATH, save_dir)
    
    save_file(new_state_dict, os.path.join(save_dir, "adapter_model.safetensors"))
    print(f"Saved TS-Mixed adapter (lambda={lambda_val}) to: {save_dir}")
    
    return save_dir

def plot_accuracies(initial_accuracies, ts_mixed_accuracies, save_path):
    """Plots a comparison of initial and TS-mixed adapter accuracies."""
    print("==> Plotting accuracies...")

    # Ensure there's data to plot
    if not initial_accuracies or not ts_mixed_accuracies:
        print("Warning: Cannot plot accuracies due to missing data.")
        return

    labels = [f"Adapter {i}" for i in initial_accuracies.keys()]
    initial_accs = list(initial_accuracies.values())
    ts_mixed_accs = list(ts_mixed_accuracies.values())

    x = torch.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, initial_accs, width, label='Initial Adapters')
    rects2 = ax.bar(x + width/2, ts_mixed_accs, width, label='TS-Mixed Adapters')

    ax.set_ylabel('Accuracy')
    ax.set_title('Adapter Accuracy Comparison: Initial vs. TS-Mixed')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')

    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Accuracy plot saved to: {save_path}")

# --- Utility Functions for Adapter Processing ---

def copy_files(src_dir, dest_dir):
    """Recursively copies files from src to dest."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def process_adapter_path(adapter_dir, num_generated):
    """Processes the specified number of newly generated adapters into the format required for testing."""
    # *** MODIFIED: Only process the newly generated adapters ***
    ckpts = [os.path.join(adapter_dir, f"{i}.safetensors") for i in range(num_generated)]

    for ckpt in ckpts:
        if not os.path.exists(ckpt):
            print(f"Warning: Checkpoint file not found, skipping: {ckpt}")
            continue

        dir_name = os.path.basename(adapter_dir) + "_" + os.path.basename(ckpt).split(".")[0]
        save_dir = os.path.join(TEST_ROOT, dir_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            # Copy base model config files
            copy_files(CONFIG_PATH, save_dir)
            # Copy and rename adapter weights
            shutil.copy2(ckpt, os.path.join(save_dir, "adapter_model.safetensors"))

# --- Main Execution ---

def main(
    eval_dataset: str, 
    test_dataset: str,
    lambda_val: float = 0.5,
    num_samples: int = None,
    inference_num_samples: int = None,
    base_adapter_path: str = None,
    base_adapter_paths: str = None
):
    """Main function to run the entire pipeline.
    
    Args:
        eval_dataset: Dataset used for evaluation/training adapters
        test_dataset: Dataset used for testing
        lambda_val: Lambda value for TS-mixing (0.0 to 1.0)
        num_samples: Optional number of samples to use for adaptation (for ablation studies)
        inference_num_samples: Optional number of samples for inference (0 = full dataset)
        base_adapter_path: Path to existing adapter to use (skip generation)
        base_adapter_paths: JSON list of adapter paths for TTS ensemble chaining
    """
    print(f"LORA Config: lambda_val={lambda_val}")
    if num_samples:
        print(f"Ablation mode: Using only {num_samples} samples")
    
    # --- Handle checkpoint chaining ---
    # If base adapter(s) provided, skip generation and use existing adapters
    if base_adapter_path or base_adapter_paths:
        print("===> Checkpoint chaining mode: Using existing adapter(s)")
        
        if base_adapter_paths:
            # TTS ensemble case - parse JSON list of adapter paths
            import json as json_module
            if isinstance(base_adapter_paths, str):
                adapter_list = json_module.loads(base_adapter_paths)
            else:
                adapter_list = base_adapter_paths
            # For each adapter, apply TS-mixing
            print(f"Applying TS-mixing to {len(adapter_list)} adapters from TTS ensemble")
            for adapter_path in adapter_list:
                adapter_name = os.path.basename(adapter_path)
                # Create a synthetic adapter_dir from the adapter path
                # The ts_mixing_lora function expects an adapter_dir with .safetensors files
                # But for existing adapters, we need to apply TS-mixing differently
                ts_mixing_lora_single(adapter_path, lambda_val)
            num_generated = len(adapter_list)
            adapter_dir = None  # Not applicable for ensemble case
        else:
            # Single adapter case
            print(f"Using base adapter: {base_adapter_path}")
            ts_mixing_lora_single(base_adapter_path, lambda_val)
            num_generated = 1
            adapter_dir = None
    else:
        # --- Original adapter generation logic ---
        # --- 1. Model and Data Loading ---
        print("==> Building model...")
        diction = torch.load(CHECKPOINT_PATH, weights_only=True, map_location="cpu")

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
        test_data_full = json.load(open(f"{COND_ROOT}/{test_dataset}_test.json", "r", encoding="utf-8"))
        if num_samples:
            test_data = test_data_full[:num_samples]
            print(f"Using {len(test_data)} samples out of {len(test_data_full)} available")
        else:
            test_data = test_data_full

        # Replicate samples for conditioning if needed (hyperconvolutional decoder requires 128)
        conditioning_data = replicate_to_min_samples(test_data)

        test_set = Dataset(
            checkpoint_folders=[f"{DATASET_ROOT}/ARC-c"],
            tokenizer=tokenizer,
            expected_iteration=None,
            real_length=config["real_length"],
            texts=[conditioning_data],
            num_texts=config["num_texts"],
            text_tokenizer=config["text_tokenizer"],
            max_text_length=config["max_text_length"],
        )

        test_loader = DataLoader(
            dataset=test_set, batch_size=1, num_workers=0,
            collate_fn=test_set.collate_fn_test, shuffle=False,
        )

        # --- 2. Generate Adapters ---
        print(f"\nGenerating adapters based on prompts from {test_dataset}")
        num_generated = generate_adapters(
            model, test_loader, test_set, eval_dataset, test_dataset, config["real_length"]
        )

        # Free up memory before inference
        del model, test_loader
        torch.cuda.empty_cache()
        gc.collect()

        if num_generated == 0:
            print("No adapters were generated. Exiting.")
            return

        # --- 3. Process Adapters ---
        print("\nProcessing generated adapters for inference...")
        adapter_dir = f"{SAVE_ROOT}/{eval_dataset}T_on_{test_dataset}V"
        # *** MODIFIED: Pass the number of generated adapters ***
        process_adapter_path(adapter_dir, num_generated)
        print("Adapter processing complete.")

        # --- 4. Implement TS-Mixing ---
        # *** MODIFIED: Pass the number of generated adapters and lambda_val ***
        ts_mixing_lora(adapter_dir, num_generated, lambda_val=lambda_val)

    # --- 5. Run Inference and Evaluate ---
    # initial_accuracies = run_inference_and_evaluate(
    #     test_dataset, num_generated, eval_dataset, test_dataset, adapter_type="basic"
    # )

    # --- 5. Run Inference and Evaluate ---
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
    ts_mixed_accuracies = run_inference_and_evaluate(
        test_dataset, num_generated, eval_dataset, test_dataset, 
        adapter_type="ts_mixed", num_samples=infer_samples, lambda_val=lambda_val
    )

    # --- 6. Plot Accuracies ---
    # if initial_accuracies and ts_mixed_accuracies:
    #     plot_save_path = f"{RES_ROOT}/{eval_dataset}T_on_{test_dataset}V_accuracy_comparison.png"
    #     plot_accuracies(initial_accuracies, ts_mixed_accuracies, plot_save_path)

    print(f"\n🎉 EXPERIMENT COMPLETED! 🎉")

if __name__ == "__main__":
    Fire(main)