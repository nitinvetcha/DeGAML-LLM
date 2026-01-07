"""
P1 - Basic Parameter Generation and Inference Script with Test-Time Learning

This script generates a specified number of LoRA adapters, prepares them for use
with a Large Language Model (LLM), performs Test-Time Learning (TTL) on each
adapter using unlabeled test data, and then evaluates each adapted adapter
individually by running inference on a test dataset and calculating the accuracy.
"""

import gc
import json
import os
import shutil
import subprocess
import sys

# --- Path and Environment Setup ---
try:
    # A more robust way to find the root directory
    root_index = os.path.abspath(__file__).split(os.sep).index("SocraticX")
    root = os.sep.join(os.path.abspath(__file__).split(os.sep)[:root_index + 1])
except ValueError:
    root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(root)
sys.path.append('/home/nitin/SocraticX/Prereqs')
os.chdir(root)
os.environ["NUM_PROCESSES"] = "1"

# --- Constants and Configuration ---
TEST_ROOT = "/home/nitin/SocraticX/Outputs/test_ckpts"
RES_ROOT = "/home/nitin/SocraticX/Outputs/results/common_sense_reasoning"
DATASET_ROOT = "/home/nitin/SocraticX/Prereqs/data/common_sense_reasoning"
CONFIG_ROOT = "/home/nitin/SocraticX/Prereqs/datasets/common_sense_reasoning"
COND_ROOT = "/home/nitin/SocraticX/Prereqs/prepare/data"
SAVE_ROOT = "/home/nitin/SocraticX/Outputs'/generated/common_sense_reasoning"
EXTRACTOR_PATH = "/home/nitin/SocraticX/Prereqs/models/all-MiniLM-L12-v2"
CONFIG_PATH = "/home/nitin/SocraticX/Prereqs/configs/Qwen0.5"
CHECKPOINT_PATH = "/home/nitin/SocraticX/Prereqs/checkpoints/qwen0.5lora__ARC-c4000.pth"
BASE_MODEL_FULL_PATH = "/home/nitin/SocraticX/Prereqs/models/Qwen2.5-0.5B-Instruct"

import torch
from fire import Fire
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.optim import AdamW

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
    "ttl_config": {
        "ttl_steps":5,
        "lr": 1e-5,
        "batch_size": 4, # Batch size for TTL updates
        "shuffle_data": True,
    }
}

# --- Core Functions ---

def generate_adapters(model, loader, dataset, dstag_T, dstag_V, real_length):
    """Generates adapters and returns the paths to the newly created checkpoint files."""
    print("==> Generating adapters...")
    model.eval()
    
    new_adapter_ckpt_paths = []
    
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

        save_dir = dataset.save_checkpoint(
            save_path=f"{SAVE_ROOT}/{dstag_T}T_on_{dstag_V}V", 
            tokens=predict[0], 
            tag=tag, 
            number=idx
        )
        
        # Construct the full path to the specific checkpoint file created
        new_ckpt_path = os.path.join(save_dir, f"{idx}.safetensors")
        new_adapter_ckpt_paths.append(new_ckpt_path)
        torch.cuda.empty_cache()

    print(f"\nSuccessfully generated {len(new_adapter_ckpt_paths)} adapters.")
    return new_adapter_ckpt_paths


def perform_ttl(adapter_path, ttl_data_loader):
    """Performs Test-Time Learning on a given adapter."""
    print(f"==> Starting Test-Time Learning for {adapter_path}...")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_FULL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    
    optimizer = AdamW(model.parameters(), lr=config["ttl_config"]["lr"])

    model.train()
    for ttl_step in range(config["ttl_config"]["ttl_steps"]):
        total_loss = 0
        num_batches = 0
        for input_ids, attention_mask in ttl_data_loader:
            optimizer.zero_grad()
            
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"TTL Step {ttl_step + 1}/{config['ttl_config']['ttl_steps']}, Average Loss: {avg_loss:.4f}")

    updated_adapter_path = f"{adapter_path}_ttl"
    model.save_pretrained(updated_adapter_path)
    print(f"✅ TTL complete. Updated adapter saved to: {updated_adapter_path}")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return updated_adapter_path


def run_inference_and_evaluate(test_dataset, adapter_paths, num_samples=None):
    """Runs inference and calculates accuracy for each adapter."""
    print(f"\n==> Starting inference and evaluation for {len(adapter_paths)} adapters...")
    
    results_dir = f"{RES_ROOT}/{test_dataset}_ttl_inference"
    os.makedirs(results_dir, exist_ok=True)
    
    BASE_MODEL_PATH_FOR_INFER = "../models/Qwen2.5-0.5B-Instruct"
    current_dir = os.getcwd()
    os.chdir("/home/nitin/SocraticX/Prereqs/prepare")
    
    accuracies = {}

    for adapter_idx, adapter_path in enumerate(adapter_paths):
        print(f"\n--- Processing Adapter {adapter_idx}/{len(adapter_paths)-1} ---")
        
        adapter_name = os.path.basename(adapter_path)
        # Include num_samples in filename to prevent caching issues
        samples_tag = f"_samples{num_samples}" if num_samples else ""
        output_file = os.path.join(results_dir, f"{adapter_name}_results{samples_tag}.jsonl")

        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter path not found, skipping: {adapter_path}")
            continue

        print(f"Running inference with adapter: {adapter_path}")
        
        try:
            infer_args = [
                "--model_name_or_path", BASE_MODEL_PATH_FOR_INFER,
                "--save_name", output_file,
                "--dataset", f"{test_dataset}_test",
                "--adapter_name_or_path", adapter_path,
                "--vllm_config", '{"gpu_memory_utilization":0.2}',
            ]
            # Add num_samples limit for ablation studies
            if num_samples:
                infer_args.extend(["--max_samples", str(num_samples)])
            subprocess.run(["python", "scripts/vllm_infer.py"] + infer_args, check=True)
            print(f"Inference complete. Results saved to: {output_file}")

            acc_args = ["--file", output_file]
            acc_result = subprocess.run(
                ["python", "scripts/calculate_acc.py"] + acc_args,
                check=True, capture_output=True, text=True
            )
            
            accuracy_line = [line for line in acc_result.stdout.split('\n') if 'acc' in line]
            if accuracy_line:
                acc_val_str = accuracy_line[0].split(':')[-1].strip()
                accuracies[adapter_idx] = float(acc_val_str)
                print(f"✅ Accuracy for adapter {adapter_idx}: {acc_val_str}")
            else:
                print(f"⚠️ Warning: Could not parse accuracy for adapter {adapter_idx}.")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during processing for adapter {adapter_idx}: {e.stderr}")
        except Exception as e:
            print(f"❌ An unexpected error occurred with adapter {adapter_idx}: {e}")

    os.chdir(current_dir) # Change back to original directory

    print("\n\n📊 ==> Evaluation Summary <==")
    if accuracies:
        avg_acc = sum(accuracies.values()) / len(accuracies)
        print(f"\nAverage Accuracy across all adapters: {avg_acc:.4f}")
    else:
        print("No accuracies were calculated.")

# --- Utility Functions for Adapter Processing ---

def copy_files(src_dir, dest_dir):
    """Recursively copies files from src to dest."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for item in os.listdir(src_dir):
        s, d = os.path.join(src_dir, item), os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def process_adapters(adapter_ckpt_paths):
    """Processes a list of specific generated adapter checkpoints for testing."""
    processed_paths = []
    for ckpt_path in adapter_ckpt_paths:
        adapter_dir_name = os.path.basename(os.path.dirname(ckpt_path))
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        
        final_dir_name = f"{adapter_dir_name}_{ckpt_name}"
        save_dir = os.path.join(TEST_ROOT, final_dir_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            copy_files(CONFIG_PATH, save_dir)
            shutil.copy2(ckpt_path, os.path.join(save_dir, "adapter_model.safetensors"))
        
        processed_paths.append(save_dir)
    return processed_paths

# --- Main Execution ---

def main(
    eval_dataset: str, 
    test_dataset: str,
    ttl_steps: int = 5,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    shuffle_data: bool = True,
    num_samples: int = None,
    inference_num_samples: int = None,
    base_adapter_path: str = None,
    base_adapter_paths: str = None
):
    """Main function to run the entire pipeline.
    
    Args:
        eval_dataset: Dataset used for evaluation/training adapters
        test_dataset: Dataset used for testing
        ttl_steps: Number of test-time learning optimization steps
        learning_rate: Learning rate for TTL optimization
        batch_size: Batch size for TTL updates
        shuffle_data: Whether to shuffle data during TTL
        num_samples: Optional number of samples to use for adaptation (for ablation studies)
        inference_num_samples: Optional number of samples for inference (0 = full dataset)
        base_adapter_path: Path to existing adapter to use (skip generation)
        base_adapter_paths: JSON list of adapter paths for TTS ensemble chaining
    """
    # Update TTL config with CLI parameters
    config["ttl_config"]["ttl_steps"] = ttl_steps
    config["ttl_config"]["lr"] = learning_rate
    config["ttl_config"]["batch_size"] = batch_size
    config["ttl_config"]["shuffle_data"] = shuffle_data
    
    print(f"TTL Config: steps={ttl_steps}, lr={learning_rate}, batch_size={batch_size}, shuffle={shuffle_data}")
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
            processed_adapter_paths = adapter_list
            print(f"Using {len(processed_adapter_paths)} adapters from TTS ensemble")
        else:
            # Single adapter case
            processed_adapter_paths = [base_adapter_path]
            print(f"Using base adapter: {base_adapter_path}")
    else:
        # --- Original adapter generation logic ---
        # --- 1. Model and Data Loading for Adapter Generation ---
        print("==> Building model...")
        diction = torch.load(CHECKPOINT_PATH, weights_only=True, map_location="cpu")

        model = Model(config=config["model_config"], criterion_weight=config["criterion_weight"].view(1, -1, 1, 1),
                      extractor_type=config["extractor_type"], extra_condition_module=config["extra_condition_module"])
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

        test_set_for_gen = Dataset(
            checkpoint_folders=[f"{DATASET_ROOT}/ARC-c"], tokenizer=tokenizer, expected_iteration=None,
            real_length=config["real_length"],
            texts=[conditioning_data],
            num_texts=config["num_texts"], text_tokenizer=config["text_tokenizer"], max_text_length=config["max_text_length"])
        
        test_loader_for_gen = DataLoader(dataset=test_set_for_gen, batch_size=1, num_workers=0,
                                         collate_fn=test_set_for_gen.collate_fn_test, shuffle=False)

        # --- 2. Generate Adapters ---
        print(f"\nGenerating adapters based on prompts from {test_dataset}")
        newly_generated_ckpt_paths = generate_adapters(
            model, test_loader_for_gen, test_set_for_gen, eval_dataset, test_dataset, config["real_length"])
        
        del model, test_loader_for_gen, test_set_for_gen
        torch.cuda.empty_cache()
        gc.collect()

        if not newly_generated_ckpt_paths:
            print("No adapters were generated. Exiting.")
            return
            
        # --- 3. Process Only the Newly Generated Adapters ---
        print("\nProcessing generated adapters for inference...")
        processed_adapter_paths = process_adapters(newly_generated_ckpt_paths)
        print("Adapter processing complete.")

    # --- 4. Prepare Data and Perform Test-Time Learning ---
    print("\n==> Preparing data for Test-Time Learning...")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_FULL_PATH, trust_remote_code=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    # Load test data for TTL (may not be loaded if using checkpoint chaining)
    test_data_full = json.load(open(f"{COND_ROOT}/{test_dataset}_test.json", "r", encoding="utf-8"))
    if num_samples:
        raw_json = test_data_full[:num_samples]
    else:
        raw_json = test_data_full
    
    try:
        raw_texts = [item['prompt'] for item in raw_json]
    except KeyError:
        raw_texts = [item['text'] for item in raw_json]

    # # --- 4. Prepare Data and Perform Test-Time Learning ---
    # print("\n==> Preparing data for Test-Time Learning...")
    # base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_FULL_PATH, trust_remote_code=True)
    # if base_tokenizer.pad_token is None:
    #     base_tokenizer.pad_token = base_tokenizer.eos_token

    # # Load the full JSON file and then slice it to get only the first 128 items
    # raw_json = json.load(open(f"{COND_ROOT}/{test_dataset}_test.json", "r", encoding="utf-8"))[:config["num_texts"]]
    # try:
    #     raw_texts = [item['prompt'] for item in raw_json]
    # except KeyError:
    #     raw_texts = [item['text'] for item in raw_json]

    ttl_inputs = base_tokenizer(raw_texts, padding=True, truncation=True, 
                                max_length=max_text_length, return_tensors="pt")

    ttl_dataset = TensorDataset(ttl_inputs['input_ids'], ttl_inputs['attention_mask'])
    ttl_loader = DataLoader(dataset=ttl_dataset, batch_size=config["ttl_config"]["batch_size"], shuffle=config["ttl_config"]["shuffle_data"])

    updated_adapter_paths = []
    for adapter_path in processed_adapter_paths:
        updated_path = perform_ttl(adapter_path, ttl_loader)
        updated_adapter_paths.append(updated_path)

    if not updated_adapter_paths:
        print("No adapters were updated with TTL. Exiting.")
        return

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
    run_inference_and_evaluate(test_dataset, updated_adapter_paths, num_samples=infer_samples)
    
    print(f"\n🎉 EXPERIMENT COMPLETED! 🎉")

if __name__ == "__main__":
    Fire(main)