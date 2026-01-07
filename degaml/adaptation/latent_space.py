"""
P2 - Basic Parameter Generation and Inference Script with SLOT Integration

This script generates LoRA adapters, computes SLOT vectors for test-time optimization,
and evaluates each adapter by running inference with SLOT applied.
"""

import gc
import json
import os
import shutil
import sys

# --- Path and Environment Setup ---
file_path = os.path.abspath(sys.argv[1])
try:
    root_index = __file__.split(os.sep).index("SocraticX")
    root = os.sep.join(__file__.split(os.sep)[:root_index + 1])
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
SAVE_ROOT = "/home/nitin/SocraticX/Outputs/generated/common_sense_reasoning"
EXTRACTOR_PATH = "/home/nitin/SocraticX/Prereqs/models/all-MiniLM-L12-v2"
CONFIG_PATH = "/home/nitin/SocraticX/Prereqs/configs/Qwen0.5"
CHECKPOINT_PATH = "/home/nitin/SocraticX/Prereqs/checkpoints/qwen0.5lora__ARC-c4000.pth"
BASE_MODEL_PATH = "/home/nitin/SocraticX/Prereqs/models/Qwen2.5-0.5B-Instruct"
SLOT_VECTORS_ROOT = "/home/nitin/SocraticX/Outputs/slot_vectors"

import torch
from fire import Fire
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

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
    "real_length": 1,
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

# --- SLOT Integration Functions ---

def compute_slot_vector(adapter_path, prompts, tokenizer, device="cuda", times=5, lr=0.1):
    """
    Computes a SLOT vector for an adapter given conditioning prompts.
    
    Args:
        adapter_path: Path to the LoRA adapter
        prompts: List of text prompts used to condition the adapter
        tokenizer: Tokenizer for the base model
        device: Device to run on
        times: Number of optimization steps
        lr: Learning rate for SLOT optimization
    
    Returns:
        SLOT delta vector (CPU tensor)
    """
    print(f"Computing SLOT vector for adapter: {adapter_path}")
    
    # Load base model with adapter
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Use a subset of prompts for efficiency (max 32)
    sample_prompts = prompts[:min(32, len(prompts))]
    
    # Format prompts with instruction template
    formatted_prompts = []
    for prompt in sample_prompts:
        formatted_prompts.append(f"Question: {prompt}\nAnswer:")
    
    # Tokenize prompts
    try:
        tokenized = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
    except Exception as e:
        print(f"Error tokenizing: {e}")
        del model, base_model
        torch.cuda.empty_cache()
        return None
    
    # Get hidden dimension
    try:
        with torch.no_grad():
            outputs = model.base_model.model(**tokenized, output_hidden_states=True)
            hidden_dim = outputs.hidden_states[-1].shape[-1]
    except Exception as e:
        print(f"Error getting hidden dim: {e}")
        del model, base_model
        torch.cuda.empty_cache()
        return None
    
    # Initialize SLOT delta vector
    delta = torch.nn.Parameter(
        0.0 * torch.randn(1, 1, hidden_dim, device=device, dtype=torch.bfloat16)
    )
    
    optimizer = torch.optim.AdamW([delta], lr=lr, weight_decay=1e-8, eps=1e-5)
    
    # Optimize delta with gradient enabled
    print(f"Optimizing SLOT vector for {times} steps...")
    for step in range(times):
        optimizer.zero_grad()
        
        with torch.enable_grad():
            # Forward pass to get hidden states
            outputs = model.base_model.model(**tokenized, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            # Apply delta to hidden states
            modified_hidden = hidden_states + delta
            
            # Get logits from LM head
            logits = model.base_model.lm_head(modified_hidden)
            
            # Compute causal language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokenized.input_ids[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            loss.backward()
            optimizer.step()
        
        if (step + 1) % 2 == 0:
            print(f"  Step {step+1}/{times}, Loss: {loss.item():.4f}")
    
    print(f"SLOT optimization complete. Final loss: {loss.item():.4f}")
    
    # Clean up and return delta
    delta_cpu = delta.detach().cpu()
    del model, base_model, delta, optimizer
    torch.cuda.empty_cache()
    
    return delta_cpu


def save_slot_vector(delta, adapter_path, slot_steps, slot_lr, num_samples=None, save_root=SLOT_VECTORS_ROOT):
    """Save SLOT vector to disk with hyperparameters and sample count in filename."""
    os.makedirs(save_root, exist_ok=True)
    adapter_name = os.path.basename(adapter_path)
    # Include hyperparameters AND num_samples in filename to avoid caching issues
    samples_tag = f"samples{num_samples}" if num_samples else "samplesfull"
    save_path = os.path.join(save_root, f"{adapter_name}_slot_steps{slot_steps}_lr{slot_lr}_{samples_tag}.pt")
    torch.save(delta, save_path)
    print(f"SLOT vector saved to: {save_path}")
    return save_path


def load_slot_vector(adapter_path, slot_steps, slot_lr, num_samples=None, save_root=SLOT_VECTORS_ROOT):
    """Load SLOT vector from disk with hyperparameters and sample count in filename."""
    adapter_name = os.path.basename(adapter_path)
    # Include hyperparameters AND num_samples in filename to avoid caching issues
    samples_tag = f"samples{num_samples}" if num_samples else "samplesfull"
    load_path = os.path.join(save_root, f"{adapter_name}_slot_steps{slot_steps}_lr{slot_lr}_{samples_tag}.pt")
    if os.path.exists(load_path):
        return torch.load(load_path, map_location="cpu", weights_only=True)
    return None


# --- Custom Inference with SLOT ---

def inference_with_slot(adapter_path, slot_vector, test_data, tokenizer, device="cuda", max_new_tokens=512):
    """
    Run inference with SLOT applied.
    
    Args:
        adapter_path: Path to LoRA adapter
        slot_vector: SLOT delta vector
        test_data: List of test examples
        tokenizer: Tokenizer
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        List of predictions
    """
    print(f"Running inference with SLOT for adapter: {adapter_path}")
    
    # Load model with adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    slot_vector = slot_vector.to(device).to(torch.bfloat16)
    
    predictions = []
    
    # Process each example
    for idx, example in enumerate(test_data):
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(test_data)} examples")
        
        prompt = example["prompt"]
        
        # Format prompt
        formatted_prompt = f"Question: {prompt}\nAnswer:"
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        # Generate with SLOT applied
        with torch.no_grad():
            # Custom generation loop with SLOT
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            generated_ids = input_ids.clone()
            past_key_values = None
            
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = model.base_model.model(
                    input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True
                )
                
                hidden_states = outputs.hidden_states[-1]
                past_key_values = outputs.past_key_values
                
                # Apply SLOT vector
                modified_hidden = hidden_states + slot_vector
                
                # Get logits
                logits = model.base_model.lm_head(modified_hidden)
                next_token_logits = logits[:, -1, :]
                
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # Append token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                ], dim=-1)
            
            # Decode
            output_text = tokenizer.decode(generated_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            predictions.append(output_text)
    
    # Clean up
    del model, base_model
    torch.cuda.empty_cache()
    
    return predictions


def calculate_accuracy(predictions, labels, dataset_name):
    """
    Calculate accuracy based on dataset type.
    """
    numbers = ["1", "2", "3", "4", "["]
    options = ["A", "B", "C", "D"]
    logic = ["True", "False", "false", "true"]
    
    def extract_answer_base(response: str):
        pattern = r"\[(.*?)\]"
        matches = re.findall(pattern, response)
        return matches
    
    correct = 0
    valid = 0
    
    for pred, label in zip(predictions, labels):
        response = pred[:20] if len(pred) >= 20 else pred
        
        if "ARC-c" in dataset_name:
            try:
                label_val = extract_answer_base(label)[0]
                pred_val = response.split("ns: ")[-1].split("\n")[0]
                if len(pred_val) > 5:
                    pred_val = [e for e in logic if e in pred_val]
                    pred_val = pred_val[0] if len(pred_val) > 0 else ""
                if pred_val in logic:
                    valid += 1
                    if pred_val.lower() == label_val.lower():
                        correct += 1
            except:
                pass
        else:
            try:
                label_val = label[1]
                if len(response) < 2:
                    continue
                
                if response[1] in numbers + options:
                    valid += 1
                    if response[1] in options:
                        pred_val = response[1]
                    elif response[1] == "[":
                        pred_val = extract_answer_base(response)[0] if extract_answer_base(response) else ""
                    else:
                        pred_val = options[numbers.index(response[1])]
                    
                    if pred_val == label_val:
                        correct += 1
            except:
                pass
    
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    validity = valid / total if total > 0 else 0
    
    return accuracy, validity


# --- Core Functions ---

def generate_adapters(model, loader, dataset, dstag_T, dstag_V, real_length, conditioning_prompts, slot_steps=5, slot_lr=0.1, num_samples=None):
    """Generates adapters and computes SLOT vectors.
    
    Args:
        slot_steps: Number of SLOT optimization steps
        slot_lr: Learning rate for SLOT optimization
        num_samples: Number of samples used for conditioning (for cache key)
    """
    print("==> Generating adapters with SLOT...")
    model.eval()
    
    adapter_paths = []
    slot_vectors = []
    
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
            
            print(f"\n==> Generating adapter {idx+1}/{real_length}...")
            predict = model(
                source=None,
                mask=mask.to(config["device"]),
                condition=conditions,
                target=None,
                generate=True,
            )

        # Save the generated adapter
        save_path = dataset.save_checkpoint(
            save_path=f"{SAVE_ROOT}/{dstag_T}T_on_{dstag_V}V", 
            tokens=predict[0], 
            tag=tag, 
            number=idx
        )
        adapter_paths.append(save_path)
        
        # Process adapter for SLOT computation
        process_adapter_path(f"{SAVE_ROOT}/{dstag_T}T_on_{dstag_V}V")
        processed_adapter_dir = f"{TEST_ROOT}/{dstag_T}T_on_{dstag_V}V_{idx}"
        
        # Compute SLOT vector
        if os.path.exists(processed_adapter_dir):
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
            slot_vec = compute_slot_vector(
                processed_adapter_dir,
                conditioning_prompts,
                tokenizer,
                device=config["device"],
                times=slot_steps,
                lr=slot_lr
            )
            
            if slot_vec is not None:
                slot_path = save_slot_vector(slot_vec, processed_adapter_dir, slot_steps, slot_lr, num_samples=num_samples)
                slot_vectors.append(slot_vec)
                print(f"✓ SLOT vector computed and saved")
            else:
                print(f"✗ Failed to compute SLOT vector")
                slot_vectors.append(None)
        else:
            print(f"✗ Processed adapter not found: {processed_adapter_dir}")
            slot_vectors.append(None)
        
        torch.cuda.empty_cache()

    print(f"\n✓ Successfully generated {len(adapter_paths)} adapters with SLOT vectors")
    return adapter_paths, slot_vectors


def run_inference_and_evaluate(test_dataset, num_adapters, dstag_T, dstag_V, num_samples=None, slot_steps=5, slot_lr=0.1, adaptation_num_samples=None):
    """Runs inference with SLOT and calculates accuracy.
    
    Args:
        num_samples: Number of samples to run inference on (None = full dataset)
        slot_steps: SLOT optimization steps (for loading correct cached vector)
        slot_lr: SLOT learning rate (for loading correct cached vector)
        adaptation_num_samples: Number of samples used during adaptation (for SLOT vector cache lookup)
    """
    print(f"\n==> Starting inference with SLOT for {num_adapters} adapters...")
    
    results_dir = f"{RES_ROOT}/{test_dataset}_slot_inference"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load test data (with optional subset for ablation)
    test_data_path = f"{COND_ROOT}/{test_dataset}_test.json"
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data_full = json.load(f)
    
    if num_samples:
        test_data = test_data_full[:num_samples]
        print(f"Ablation mode: Using {len(test_data)} samples out of {len(test_data_full)} available")
    else:
        test_data = test_data_full
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    accuracies = {}

    for adapter_idx in range(num_adapters):
        print(f"\n{'='*60}")
        print(f"Processing Adapter {adapter_idx}/{num_adapters-1}")
        print(f"{'='*60}")
        
        adapter_path = f"{TEST_ROOT}/{dstag_T}T_on_{test_dataset}V_{adapter_idx}"
        
        if not os.path.exists(adapter_path):
            print(f"⚠️  Adapter not found: {adapter_path}")
            continue
        
        # Load SLOT vector using adaptation_num_samples (the sample count used during SLOT computation)
        # This is crucial when adaptation uses a subset but inference runs on full dataset
        cache_num_samples = adaptation_num_samples if adaptation_num_samples is not None else num_samples
        slot_vector = load_slot_vector(adapter_path, slot_steps, slot_lr, num_samples=cache_num_samples)
        if slot_vector is None:
            print(f"⚠️  SLOT vector not found for steps={slot_steps}, lr={slot_lr}, samples={cache_num_samples}, skipping adapter {adapter_idx}")
            continue
        
        print(f"✓ Loaded SLOT vector")
        
        # Run inference with SLOT
        try:
            predictions = inference_with_slot(
                adapter_path,
                slot_vector,
                test_data,
                tokenizer,
                device=config["device"],
                max_new_tokens=128
            )
            
            # Extract labels
            labels = [item.get("response", "") for item in test_data]           
             
            # Calculate accuracy
            accuracy, validity = calculate_accuracy(predictions, labels, test_dataset)
            accuracies[adapter_idx] = accuracy
            
            print(f"✅ Adapter {adapter_idx} - Accuracy: {accuracy:.4f}, Validity: {validity:.4f}")
            
            # Save results in JSONL format (like P1.py)
            # Include hyperparameters AND num_samples in filename to prevent caching issues
            samples_tag = f"samples{num_samples}" if num_samples else "samplesfull"
            output_file = f"{results_dir}/{dstag_T}T_on_{test_dataset}V_adapter_{adapter_idx}_slot_steps{slot_steps}_lr{slot_lr}_{samples_tag}.jsonl"
            
            print(f"Saving results to JSONL file: {output_file}")
            
            with open(output_file, "w") as f:
                # Write each prediction/label pair as a separate line
                for pred, label in zip(predictions, labels):
                # The predict/label keys and string formatting mimic P1's external script output
                    line_data = {
                        "predict": f" {pred.strip()}",
                        "label": f"{label}<|im_end|>\n"
                    }
                    f.write(json.dumps(line_data) + "\n")
      
            # Note: The overall accuracy/validity calculation is now separate and might be printed only.
            
        except Exception as e:
            print(f"❌ Error processing adapter {adapter_idx}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "="*60)
    print("📊 EVALUATION SUMMARY (with SLOT)")
    print("="*60)
    if accuracies:
        for idx, acc in sorted(accuracies.items()):
            print(f"Adapter {idx}: Accuracy = {acc:.4f}")
        avg_acc = sum(accuracies.values()) / len(accuracies)
        print(f"\n🎯 Average Accuracy: {avg_acc:.4f}")
        print("="*60)
    else:
        print("No accuracies were calculated.")


# --- Utility Functions ---

def find_safetensors_files(directory):
    """Finds all .safetensors files in a directory."""
    safetensors_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".safetensors"):
                safetensors_files.append(os.path.join(root, file))
    return safetensors_files


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


def process_adapter_path(adapter_dir):
    """Processes generated adapters into the format required for testing."""
    ckpts = find_safetensors_files(adapter_dir)
    for ckpt in ckpts:
        dir_name = adapter_dir.split("/")[-1] + "_" + os.path.basename(ckpt).split(".")[0]
        save_dir = os.path.join(TEST_ROOT, dir_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            copy_files(CONFIG_PATH, save_dir)
            shutil.copy2(ckpt, os.path.join(save_dir, "adapter_model.safetensors"))


# --- Main Execution ---

def main(
    eval_dataset: str, 
    test_dataset: str,
    slot_steps: int = 5,
    slot_lr: float = 0.1,
    num_samples: int = None,
    inference_num_samples: int = None,
    base_adapter_path: str = None,
    base_adapter_paths: str = None
):
    """Main function to run the entire pipeline with SLOT.
    
    Args:
        eval_dataset: Dataset used for evaluation/training adapters
        test_dataset: Dataset used for testing
        slot_steps: Number of SLOT optimization steps
        slot_lr: Learning rate for SLOT optimization
        num_samples: Optional number of samples to use for adaptation (for ablation studies)
        inference_num_samples: Optional number of samples for inference (0 = full dataset)
        base_adapter_path: Path to existing adapter to use (skip generation)
        base_adapter_paths: JSON list of adapter paths for TTS ensemble chaining
    """
    print(f"Latent Config: slot_steps={slot_steps}, slot_lr={slot_lr}")
    if num_samples:
        print(f"Ablation mode: Using only {num_samples} samples")
    print("\n" + "="*60)
    print("P1 with SLOT Integration")
    print("="*60 + "\n")
    
    # Load conditioning prompts (needed for SLOT computation in all cases)
    test_prompts_path = f"{COND_ROOT}/{test_dataset}_test.json"
    with open(test_prompts_path, "r", encoding="utf-8") as f:
        test_prompts_data_full = json.load(f)
    
    if num_samples:
        test_prompts_data = test_prompts_data_full[:num_samples]
        print(f"Using {len(test_prompts_data)} samples out of {len(test_prompts_data_full)} available")
    else:
        test_prompts_data = test_prompts_data_full
    
    # Replicate samples for conditioning if needed (hyperconvolutional decoder requires 128)
    conditioning_data = replicate_to_min_samples(test_prompts_data)
    
    # Extract prompts for SLOT conditioning
    try:
        conditioning_prompts = [item["prompt"] for item in conditioning_data]
    except:
        conditioning_prompts = [item["conversations"][0]["value"] for item in conditioning_data]
    
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
            print(f"Computing SLOT for {len(adapter_list)} adapters from TTS ensemble")
        else:
            # Single adapter case
            adapter_list = [base_adapter_path]
            print(f"Using base adapter: {base_adapter_path}")
        
        # Compute SLOT for each existing adapter
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        adapter_paths = []
        
        for adapter_path in adapter_list:
            if not os.path.exists(adapter_path):
                print(f"Warning: Adapter not found: {adapter_path}")
                continue
            
            adapter_paths.append(adapter_path)
            
            # Compute new SLOT vector for this adapter
            slot_vec = compute_slot_vector(
                adapter_path,
                conditioning_prompts,
                tokenizer,
                device=config["device"],
                times=slot_steps,
                lr=slot_lr
            )
            
            if slot_vec is not None:
                save_slot_vector(slot_vec, adapter_path, slot_steps, slot_lr, num_samples=num_samples)
                print(f"✓ SLOT vector computed and saved for {os.path.basename(adapter_path)}")
            else:
                print(f"✗ Failed to compute SLOT vector for {os.path.basename(adapter_path)}")
            
            torch.cuda.empty_cache()
        
        if not adapter_paths:
            print("❌ No valid adapters found. Exiting.")
            return
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
        tokenizer_obj = Tokenizer(token_size=config["token_size"])
        model.load_state_dict(diction, strict=False)
        model.to(config["device"])

        test_set = Dataset(
            checkpoint_folders=[f"{DATASET_ROOT}/ARC-c"],
            tokenizer=tokenizer_obj,
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

        # --- 2. Generate Adapters with SLOT ---
        print(f"\n==> Generating adapters with SLOT based on prompts from {test_dataset}")
        adapter_paths, slot_vectors = generate_adapters(
            model, test_loader, test_set, eval_dataset, test_dataset,
            config["real_length"], conditioning_prompts,
            slot_steps=slot_steps, slot_lr=slot_lr, num_samples=num_samples
        )
        
        # Free up memory
        del model, test_loader
        torch.cuda.empty_cache()
        gc.collect()

        if not adapter_paths:
            print("❌ No adapters were generated. Exiting.")
            return

    # --- 3. Run Inference with SLOT and Evaluate ---
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
        test_dataset, len(adapter_paths), eval_dataset, test_dataset, 
        num_samples=infer_samples, slot_steps=slot_steps, slot_lr=slot_lr,
        adaptation_num_samples=num_samples  # Pass original num_samples for SLOT cache lookup
    )
    
    print(f"\n{'='*60}")
    print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY! 🎉")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    Fire(main)