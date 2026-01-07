# -*- coding: utf-8 -*-
"""
P1 - Unified script for Adapter Generation, Confidence-based Inference, and Evaluation.

This script combines the functionality of the original three scripts:
1.  MODIFICATION: Generates 20 LoRA adapters using a HyperConvDecoderModel by feeding
    the same randomly sampled prompt batch to the generator 20 times.
2.  Performs inference on a test dataset using each of the 20 generated adapters
    independently with a vLLM-powered model. It calculates and stores a confidence
    score (log probability) for each prediction.
3.  Aggregates the results from all adapters. For each question, it selects the
    prediction determined by a majority vote across all 20 inference runs.
"""

# =================================================================================================
# SECTION 1: IMPORTS (Consolidated from all three original scripts)
# =================================================================================================
import gc
import json
import multiprocessing as mp
import os
import random # MODIFICATION: Added for prompt sampling
import re
import shutil
import sys
import threading
import time
from collections import Counter
from typing import Optional

import accelerate.utils
import fire
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, Seq2SeqTrainingArguments

# --- LlamaFactory and vLLM Imports (from vllm_infer.py) ---
# Ensure LlamaFactory is in the Python path
# Assuming LlamaFactory is installed and accessible
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import check_version, get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
else:
    raise ImportError("vLLM is not available. Please install it to run this script.")

# --- Custom Model/Dataset Imports (from P1.py) ---
# Add prerequisite paths
# Please adjust these paths if your project structure is different
sys.path.append("/home/nitin/SocraticX/Prereqs")
from degaml.generator.dataset import Text2Qwen25LoRA_FullCondDataset as Dataset
from degaml.generator.model import HyperConvDecoderModel_FullCond as Model
from degaml.generator.tokenizer import Qwen2505LoRA_Tokenizer2D as Tokenizer


# =================================================================================================
# SECTION 2: CONSTANTS AND CONFIGURATION (from P1.py)
# =================================================================================================
# --- Path Configurations ---
# It's recommended to use absolute paths to avoid issues with os.chdir
TEST_ROOT = "/home/nitin/SocraticX/Outputs/test_ckpts"
RES_ROOT = "/home/nitin/SocraticX/Outputs/results_sing_diff/common_sense_reasoning"
DATASET_ROOT = "/home/nitin/SocraticX/Prereqs/data/common_sense_reasoning"
CONFIG_ROOT = "/home/nitin/SocraticX/Prereqs/datasets/common_sense_reasoning"
COND_ROOT = "/home/nitin/SocraticX/Prereqs/prepare/data"
SAVE_ROOT = "/home/nitin/SocraticX/Outputs/generated/common_sense_reasoning"
EXTRACTOR_PATH = "/home/nitin/SocraticX/Prereqs/models/all-MiniLM-L12-v2"
CONFIG_PATH = "/home/nitin/SocraticX/Prereqs/configs/Qwen0.5"
BASE_MODEL_PATH = "../models/Qwen2.5-0.5B-Instruct" # Relative to the `prepare` dir
PREPARE_DIR = "/home/nitin/SocraticX/Prereqs/prepare"


# --- General Configurations ---
SEED = 999
torch.set_float32_matmul_precision("high")
accelerate.utils.set_seed(SEED)
import math

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
dataset_tag = "ARC-c"  # Example dataset tag

config = {
    # global setting
    "need_test": False,
    # data setting
    "token_size": (10, 130),
    "real_length": 3,  # MODIFICATION: Number of adapters to generate / inference runs
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
    # generate setting
    "device": "cuda",
    "num_generated": 10,
}

# =================================================================================================
# SECTION 3: ACCURACY CALCULATION LOGIC (from calculate_acc.py)
# =================================================================================================
# --- Constants for answer parsing ---
numbers = ["1", "2", "3", "4", "["]
options = ["A", "B", "C", "D"]
logic = ["True", "False", "false", "true"]

def extract_answer_base(response: str):
    """Extracts answers enclosed in square brackets."""
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, response)
    return matches

def extract_final_choice(prediction_text: str):
    """
    Extracts the first valid choice (A, B, C, D, True, False) from the prediction text.
    Prioritizes choices inside brackets.
    """
    # Look for a choice inside one or more brackets, case-insensitive
    bracket_match = re.search(r'\[+\s*(A|B|C|D|T|F|True|False)', prediction_text, re.IGNORECASE)
    if bracket_match:
        choice = bracket_match.group(1).upper()
        # Normalize to 'True' or 'False' for boolean questions
        if choice == 'T': return 'True'
        if choice == 'F': return 'False'
        if choice == 'TRUE': return 'True'
        if choice == 'FALSE': return 'False'
        return choice

    # If no bracketed answer, look for the first occurrence of True/False (as whole words)
    bool_match = re.search(r'\b(True|False)\b', prediction_text, re.IGNORECASE)
    if bool_match:
        return bool_match.group(1).capitalize()

    # If still no match, look for the first occurrence of A/B/C/D as a character
    char_match = re.search(r'\b(A|B|C|D)\b', prediction_text)
    if char_match:
        return char_match.group(1)

    return None # Return None if no valid choice is found

def check_validity(file: str):
    """
    Calculates the validity and accuracy of predictions in a given file.
    """
    print(f"\n--- Calculating Accuracy for: {os.path.basename(file)} ---")
    try:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {file}")
        return

    count, valid, acc = 0, 0.0, 0.0
    for line in lines:
        if not line.strip():
            continue
        count += 1
        answer_dict = json.loads(line.strip())
        response = answer_dict["predict"][:20]

        if "ARC-c" in file:
            label = extract_answer_base(answer_dict["label"])[0]
            pred_parts = response.split("ns: ")[-1].split("\n")
            pred = pred_parts[0] if pred_parts else ""
            if len(pred) > 5:
                pred_options = [e for e in logic if e in pred]
                pred = pred_options[0] if pred_options else ""
            if pred in logic:
                valid += 1
                if pred.lower() == label.lower():
                    acc += 1
        else:
            # The label is the character inside the brackets, e.g., 'C' from '[C]<|im_end|>\n'
            label_match = re.search(r'\[(A|B|C|D)\]', answer_dict["label"])
            if not label_match:
                continue
            label = label_match.group(1)

            pred = ""
            # Extract the first character A, B, C, or D from the prediction
            pred_match = re.search(r'(A|B|C|D)', response)
            if pred_match:
                valid += 1
                pred = pred_match.group(1)
                if pred == label:
                    acc += 1
    
    if count == 0:
        print("No predictions found in the file.")
        return

    valid_ratio = valid / count
    acc_ratio = acc / count
    print(f"Total Questions: {count}")
    print(f"Valid Predictions: {valid_ratio:.4f} ({int(valid)}/{count})")
    print(f"Final Accuracy: {acc_ratio:.4f} ({int(acc)}/{count})")
    print("--------------------------------------------------")


# =================================================================================================
# SECTION 4: VLLM INFERENCE LOGIC (from vllm_infer.py)
# =================================================================================================

def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: int = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
):
    """
    Performs batch generation using vLLM engine, modified to output confidence scores for all choices.
    """
    check_version("vllm>=0.4.3")
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of GPUs.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    inputs, labels = [], []
    for sample in dataset_module["train_dataset"]:
        multi_modal_data = {"image": template_obj.mm_plugin._regularize_images(
            sample["images"], image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
        )} if sample["images"] else None
        inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})
        labels.append(tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=False))

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,
        top_k=generating_args.top_k,
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=False,
        seed=seed,
        logprobs=15,  # Request log probabilities for top tokens to find choices
    )
    
    lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0]) if model_args.adapter_name_or_path else None
    
    engine_args = {
        "model": model_args.model_name_or_path, "trust_remote_code": True,
        "dtype": model_args.infer_dtype, "tensor_parallel_size": 2, # As in original script
        "pipeline_parallel_size": pipeline_parallel_size, "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None, "max_lora_rank": 64,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2}
    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    llm_engine = LLM(**engine_args)
    results = llm_engine.generate(inputs, sampling_params, lora_request=lora_request)
    
    # Process results to extract predictions and choice log probabilities
    output_data = []
    for result in results:
        output = result.outputs[0]
        prediction_text = output.text
        
        all_possible_choices = {"A", "B", "C", "D", "True", "False", "T", "F"}
        choice_logprobs = {
            "A": -float('inf'), "B": -float('inf'), "C": -float('inf'), "D": -float('inf'),
            "True": -float('inf'), "False": -float('inf'), "T": -float('inf'), "F": -float('inf')
        }
        logprobs_list = output.logprobs
        
        # CORRECTED LOGIC: Use a history to robustly find the decision point.
        if logprobs_list:
            # A list of the top decoded tokens generated so far
            history_tokens = []
            for logprob_dict in logprobs_list:
                if not logprob_dict:
                    continue
                
                top_token_str = list(logprob_dict.values())[0].decoded_token
                
                # Check if the current token is a potential choice
                if top_token_str.strip() in all_possible_choices:
                    # If it is, check if a bracket has appeared in the history
                    full_history = "".join(history_tokens)
                    if '[' in full_history:
                        # This is our decision point. Capture logprobs and exit.
                        for logprob_obj in logprob_dict.values():
                            choice_token = logprob_obj.decoded_token.strip()
                            if choice_token in all_possible_choices:
                                choice_logprobs[choice_token] = logprob_obj.logprob
                        break
                
                # Add the current token to the history for the next step's check
                history_tokens.append(top_token_str)

        # Determine the primary confidence score based on the actual predicted choice.
        # The regex still finds a choice following ONE OR MORE brackets (`\[+`).
        pred_match = re.search(r'\[+\s*(A|B|C|D|True|False|T|F)', prediction_text, re.IGNORECASE)
        
        predicted_choice_key = None
        if pred_match:
            raw_choice = pred_match.group(1)
            # Find the correctly-cased key from our dictionary to ensure a match.
            for key in choice_logprobs:
                if key.lower() == raw_choice.lower():
                    predicted_choice_key = key
                    break
        
        confidence = choice_logprobs.get( predicted_choice_key, -float('inf'))

        output_data.append({
            "predict": prediction_text,
            "confidence": confidence,
            "choice_logprobs": choice_logprobs
        })

    # Save results to file
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    with open(save_name, "w", encoding="utf-8") as f:
        for i, data in enumerate(output_data):
            f.write(json.dumps({
                "predict": data["predict"],
                "label": labels[i],
                "confidence": data["confidence"],
                "choice_logprobs": data["choice_logprobs"]
            }, ensure_ascii=False) + "\n")

    print(f"--> {len(output_data)} generated results with confidence scores have been saved at {os.path.basename(save_name)}")
    del llm_engine
    gc.collect()
    torch.cuda.empty_cache()


# =================================================================================================
# SECTION 5: ADAPTER GENERATION AND ORCHESTRATION LOGIC (from P1.py)
# =================================================================================================
def generate(model, loader, dataset, dstag_T, dstag_V):
    """Generates adapter checkpoints."""
    print("\n==> Generating Adapters...")
    model.eval()
    for idx, (tokens, cond_id, cond_mask, tag) in enumerate(loader):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mask = ~torch.isnan(tokens)
            tokens = torch.nan_to_num_(tokens, nan=0.0)
            conditions = {
                "input_ids": cond_id.to(device=config["device"]),
                "attention_mask": cond_mask.to(device=config["device"]),
            }
            predict = model(source=None, mask=mask.to(config["device"]), condition=conditions, target=None, generate=True)
        
        print(f"Generated adapter {idx + 1}/{len(loader)}...")
        dataset.save_checkpoint(
            save_path=f"{SAVE_ROOT}/{dstag_T}T_on_{dstag_V}V", tokens=predict[0], tag=tag, number=idx
        )
        torch.cuda.empty_cache()

def find_safetensors_files(directory):
    """Finds all .safetensors files in a directory."""
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(".safetensors")]

def copy_files(src_dir, dest_dir):
    """Recursively copies files from source to destination."""
    os.makedirs(dest_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def process_adapter_path(adapter_dir):
    """Processes generated adapters into the format expected by the inference script."""
    print("==> Processing generated adapters for inference...")
    ckpts = find_safetensors_files(adapter_dir)
    for ckpt in ckpts:
        dir_name = os.path.basename(adapter_dir) + "_" + os.path.basename(ckpt).split(".")[0]
        save_dir = os.path.join(TEST_ROOT, dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            copy_files(CONFIG_PATH, save_dir)
            shutil.copy2(ckpt, os.path.join(save_dir, "adapter_model.safetensors"))
    print(f"Processed {len(ckpts)} adapters.")


def main(
    eval_dataset: str, 
    test_dataset: str,
    num_lora_adpt: int = 3,
    method: str = "majority_vote",
    num_samples: int = None,
    inference_num_samples: int = None
):
    """
    Main orchestration function.
    
    Args:
        eval_dataset: Dataset used for evaluation/training adapters
        test_dataset: Dataset used for testing
        num_lora_adpt: Number of LoRA adapters to generate
        method: Aggregation method - 'majority_vote' or 'max_confidence'
        num_samples: Optional number of samples to use for adaptation (for ablation studies)
        inference_num_samples: Optional number of samples for inference (0 = full dataset)
    """
    # Update config with CLI parameters
    config["real_length"] = num_lora_adpt
    
    print(f"TTS Config: num_lora_adpt={num_lora_adpt}, method={method}")
    if num_samples:
        print(f"Ablation mode: Using only {num_samples} samples")
    # FIX: Set multiprocessing start method to 'spawn' to prevent CUDA conflicts
    # This is crucial for running vLLM in a loop.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # context has already been set

    # --- Part 1: Generate 20 Adapters from a single sampled prompt batch ---
    print("==> Building adapter generation model...")
    diction = torch.load(f"/home/nitin/SocraticX/Prereqs/checkpoints/qwen0.5lora__ARC-c4000.pth", weights_only=True, map_location="cpu")
    model = Model(
        config=config["model_config"],
        criterion_weight=config["criterion_weight"].view(1, -1, 1, 1),
        extractor_type=config["extractor_type"],
        extra_condition_module=config["extra_condition_module"],
    )
    tokenizer = Tokenizer(token_size=config["token_size"])
    model.load_state_dict(diction, strict=False)
    model.to(config["device"])

    # MODIFICATION: Randomly sample a single batch of prompts to be used for all generations
    all_texts = json.load(open(f"{COND_ROOT}/{test_dataset}_test.json", "r", encoding="utf-8"))
    
    # Apply num_samples limit for ablation studies
    if num_samples:
        all_texts = all_texts[:num_samples]
        print(f"Using {len(all_texts)} samples for ablation study")
    
    # Replicate samples for conditioning if needed (hyperconvolutional decoder requires 128)
    conditioning_data = replicate_to_min_samples(all_texts)

    random.seed(SEED) # for reproducibility
    if len(conditioning_data) > config["num_texts"]:
        sampled_prompt_batch = random.sample(conditioning_data, config["num_texts"])
    else:
        sampled_prompt_batch = conditioning_data # Use all if not enough to sample from
    print(f"==> Sampled {len(sampled_prompt_batch)} prompts to be used as the condition for all adapter generations.")

    test_set = Dataset(
        checkpoint_folders=[f"{DATASET_ROOT}/ARC-c"], tokenizer=tokenizer, expected_iteration=None,
        real_length=config["real_length"],      # MODIFICATION: Generate 20 adapters
        texts=[sampled_prompt_batch],           # MODIFICATION: Use the same sampled batch for all generations
        num_texts=config["num_texts"], text_tokenizer=config["text_tokenizer"],
        max_text_length=config["max_text_length"],
    )
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=0, collate_fn=test_set.collate_fn_test, shuffle=False)
    
    generate(model, test_loader, test_set, eval_dataset, test_dataset)
    del model, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    process_adapter_path(f"{SAVE_ROOT}/{eval_dataset}T_on_{test_dataset}V")
    
    # --- Part 2: Run Inference, once for each of the 20 generated adapters ---
    print(f"\n==> Starting inference for {config['real_length']} generated adapters...")
    original_cwd = os.getcwd()
    os.chdir(PREPARE_DIR)  # Change to dir with relative paths for models/data

    # MODIFICATION: Loop through each of the 20 generated adapters
    for i in range(config["real_length"]):
        adapter_path = f"{TEST_ROOT}/{eval_dataset}T_on_{test_dataset}V_{i}"
        print(f"\n--- Running Inference Run {i + 1}/{config['real_length']} with Adapter: {os.path.basename(adapter_path)} ---")
        
        if not os.path.exists(adapter_path):
            print(f"WARNING: Adapter not found at {adapter_path}. Skipping.")
            continue

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
        
        inference_kwargs = {
            "model_name_or_path": BASE_MODEL_PATH,
            # Include infer_samples in filename to prevent caching issues
            "save_name": f"{RES_ROOT}/{test_dataset}/{eval_dataset}T_on_{test_dataset}V_{i}_samples{infer_samples if infer_samples else 'full'}.jsonl", # Save name is unique per run
            "dataset": f"{test_dataset}_test",
            "adapter_name_or_path": adapter_path, # MODIFICATION: Use the i-th adapter
            "vllm_config": '{"gpu_memory_utilization":0.5}',
            "max_samples": infer_samples,  # Pass inference_num_samples for ablation
        }
        
        process = mp.Process(target=vllm_infer, kwargs=inference_kwargs)
        process.start()
        process.join()

        if process.exitcode != 0:
            print(f"ERROR: Inference process for adapter {i} failed with exit code {process.exitcode}.")
    
    os.chdir(original_cwd)

    # --- Part 3: Aggregate Results Based on Majority Vote ---
    print("\n==> Aggregating results based on majority voting...")
    samples_suffix = f"_samples{infer_samples if infer_samples else 'full'}"
    prediction_files = [f"{RES_ROOT}/{test_dataset}/{eval_dataset}T_on_{test_dataset}V_{i}{samples_suffix}.jsonl" for i in range(config["real_length"])]
    valid_files = [f for f in prediction_files if os.path.exists(f)]

    if not valid_files:
        print("Error: No prediction files were generated. Exiting.")
        return
    
    try:
        with open(valid_files[0], 'r', encoding='utf-8') as f:
            num_questions = sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"Error reading first prediction file to get question count: {e}")
        return

    if num_questions == 0:
        print("Error: First prediction file is empty. Cannot aggregate results.")
        return

    file_handles = [open(f, 'r', encoding='utf-8') for f in valid_files]
    final_predictions = []

    for q_idx in range(num_questions):
        question_predictions = []
        prediction_data_map = {} # To store a sample record for each choice

        for handle in file_handles:
            line = handle.readline()
            if not line: continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            choice = extract_final_choice(data.get("predict", ""))
            if choice:
                question_predictions.append(choice)
                # Store the first full data record we see for this choice
                if choice not in prediction_data_map:
                    prediction_data_map[choice] = data

        if not question_predictions:
            print(f"Warning: No valid choices found for question index {q_idx}. Skipping.")
            continue

        # Aggregate based on method
        if method == "max_confidence":
            # Select prediction with highest confidence score
            best_confidence = -float('inf')
            best_data = None
            for handle_idx, handle in enumerate(file_handles):
                pass  # Already read above, use prediction_data_map
            
            # Find the prediction with max confidence from prediction_data_map
            for choice, data in prediction_data_map.items():
                conf = data.get("confidence", -float('inf'))
                if conf > best_confidence:
                    best_confidence = conf
                    best_data = data
            
            if best_data:
                final_predictions.append({
                    "predict": best_data["predict"],
                    "label": best_data["label"]
                })
        else:
            # Default: majority voting
            votes = Counter(question_predictions)
            # In case of a tie, most_common picks one arbitrarily, which is fine.
            majority_choice, _ = votes.most_common(1)[0]
            
            # Get a representative record for the majority choice
            if majority_choice in prediction_data_map:
                best_prediction_data = prediction_data_map[majority_choice]
                final_predictions.append({
                    "predict": best_prediction_data["predict"],
                    "label": best_prediction_data["label"]
                })
            else:
                 print(f"Warning: Majority choice '{majority_choice}' not found in data map for question {q_idx}. This is unexpected.")

    for handle in file_handles:
        handle.close()

    # --- Part 4: Save Final Predictions and Calculate Final Accuracy ---
    final_output_path = f"{RES_ROOT}/{test_dataset}/{eval_dataset}T_on_{test_dataset}V_final{samples_suffix}.jsonl"
    with open(final_output_path, "w", encoding="utf-8") as f:
        for item in final_predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n==> Aggregation complete. Final predictions saved to:\n{final_output_path}")

    check_validity(file=final_output_path)
    print("\n==> Evaluation finished successfully! ==")


# =================================================================================================
# SECTION 6: SCRIPT ENTRY POINT
# =================================================================================================
if __name__ == "__main__":
    fire.Fire(main)