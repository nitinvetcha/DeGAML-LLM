#!/usr/bin/env python3
"""
Hypothesis Generation Module for DeGAML-LLM

This module generates adaptation strategy hypotheses using a Large Language Model (LLM).
It proposes different adaptation configurations across four families:
- TTT (Test-Time Training): Fine-tune adapters on test data
- TTS (Test-Time Scaling): Ensemble multiple adapters
- LORA (LoRA Mixing): Interpolate between LoRA subspaces
- Latent (SLOT): Optimize sample-specific latent vectors

The module uses vLLM for efficient hypothesis generation and supports:
- Family-specific filtering for ablation studies
- Dynamic prompt shuffling to avoid mode collapse
- Constraint-based generation (minimum samples per family)
- Optional policy adapter for hypothesis refinement

Usage:
    python -m degaml.core.hypothesis_generation \\
        --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \\
        --lora_adapter_path ./checkpoints/policy_adapter \\
        --num_generations 20 \\
        --output_file ./outputs/hypotheses.txt
    
    # For ablation studies (single family):
    python -m degaml.core.hypothesis_generation \\
        --family_filter TTT \\
        --num_generations 5
"""

import fire
import json
import os
import random
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

# vLLM imports
try:
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams.data_args import DataArguments
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    raise ImportError(
        "Required packages are not installed. "
        "Please install: pip install vllm transformers torch fire llamafactory tqdm"
    )

from degaml.utils.paths import PathConfig

# Initialize path configuration
paths = PathConfig()


# Prompt template components
PROMPT_HEADER = """
You are an expert AI agent tasked with generating an optimal adaptation strategy for a Large Language Model to improve its performance on a new, unseen task. Your goal is to output a single, structured JSON object that specifies the most promising strategy.

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
"""

# The 4 option blocks (without numbers)
OPTION_BLOCKS = [
    # TTT
    """If you choose the Test-Time Training family, use this format:

    ```json
    {
      "family": "TTT",
      "ttl_steps": <integer>,
      "learning_rate": <float>,
      "batch_size": <integer>,
      "shuffle_data": <boolean>
    }
    ```

    * Method Description: This strategy updates the model's weights at inference time using only unlabeled test data. Specifically, it performs self-supervised enhancement by minimizing the input perplexity.""",

    # LORA
    """If you choose the LoRA Modification family, use this format:

    ```json
    {
      "family": "LORA",
      "lambda": <float>
    }
    ```

    * Method Description: This strategy uses the two-subspace (TS) mixing approach. Instead of the standard LoRA update ($A_1B_1 + A_2B_2$), this method introduces extra cross-term interactions by computing $(A_1+A_2)(B_1+B_2)$ to mix the two LoRA subspaces. The `lambda` hyperparameter determines the mixing ratio.""",

    # TTS
    """If you choose the Test-Time Scaling family, use this format:

    ```json
    {
      "family": "TTS",
      "num_lora_adpt": <integer>,
      "method": "<string>"
    }
    ```

    * Method Description: This strategy involves generating multiple LoRA adapters, running inference with all of them, and then aggregating the predictions to get a final answer, for instance, by selecting the most confident prediction or using a majority vote.
    * The "method" key should contain one of the two specific string values - `"max_confidence"`, `"majority_vote"`.""",

    # Latent
    """If you choose the Latent Space Modification family, use this format:

    ```json
    {
      "family": "Latent",
      "slot_steps": <integer>,
      "slot_lr": <float>
    }
    ```

    * Method Description: This strategy utilizes Sample-specific Language Model Optimization at Test-time (SLOT). It optimizes a lightweight, sample-specific parameter vector (delta) added to the final hidden layer features. By minimizing the cross-entropy loss on the input prompt itself over `slot_steps` (typically 1-5) with a learning rate of `slot_lr`, it aligns the model's internal representation to the specific instruction before generating the response."""
]

PROMPT_FOOTER = "\nNow, please provide only the JSON object for the ARC-c dataset."

# Mapping from family name to option block index
FAMILY_TO_BLOCK_IDX = {
    "TTT": 0,
    "LORA": 1,
    "TTS": 2,
    "Latent": 3,
}


def generate_from_prompt(
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    lora_adapter_path: str = None,
    template: str = "qwen",
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.05,
    num_generations: int = 20,
    output_file: str = None,
    family_filter: str = None,
):
    """Generate adaptation strategy hypotheses using an LLM.
    
    This function performs iterative generation to collect diverse JSON outputs
    representing different adaptation strategies. It uses dynamic prompt shuffling
    to ensure diversity across method families.
    
    Args:
        model_name_or_path: Path or HuggingFace ID of the base LLM
        lora_adapter_path: Optional path to LoRA adapter (policy adapter)
        template: Chat template name (default: "qwen")
        temperature: Sampling temperature for diversity (default: 1.0)
        top_p: Nucleus sampling parameter (default: 0.9)
        top_k: Top-k sampling parameter (default: 50)
        max_new_tokens: Maximum tokens to generate per sample (default: 512)
        repetition_penalty: Repetition penalty (default: 1.05)
        num_generations: Target number of unique hypotheses (default: 20)
        output_file: Path to save generated hypotheses (default: outputs/hypotheses.txt)
        family_filter: Optionally filter to single family for ablation (e.g., "TTT")
    """
    # Set default output path
    if output_file is None:
        output_file = str(paths.output_root / "hypotheses" / "generation.txt")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Filter to single family if specified
    active_families = list(FAMILY_TO_BLOCK_IDX.keys())
    option_blocks = OPTION_BLOCKS.copy()
    
    if family_filter:
        # Handle case-insensitive matching
        family_map = {k.upper(): k for k in FAMILY_TO_BLOCK_IDX.keys()}
        family_map["LATENT"] = "Latent"  # Special case
        
        if family_filter.upper() in family_map:
            target_family = family_map[family_filter.upper()]
            block_idx = FAMILY_TO_BLOCK_IDX[target_family]
            option_blocks = [OPTION_BLOCKS[block_idx]]
            active_families = [target_family]
            print(f"🎯 Family filter active: Only generating '{target_family}' hypotheses")
        else:
            print(f"⚠️  Unknown family filter '{family_filter}'. Valid options: {list(FAMILY_TO_BLOCK_IDX.keys())}")
            print("Proceeding with all families.")
    
    print(f"Using base model: {model_name_or_path}")
    print("-" * 70)
    
    # --- Initialize Tokenizer & Engine ---
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    data_args = DataArguments(template=template)
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # Set up generation parameters
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        skip_special_tokens=True,
    )
    
    use_lora = False
    if lora_adapter_path and os.path.exists(lora_adapter_path):
        print(f"✅ LoRA adapter found: {lora_adapter_path}. Enabling LoRA in vLLM.")
        use_lora = True
    else:
        print("ℹ️  No valid LoRA adapter path provided. Using the base model only.")
    
    print("Initializing vLLM engine...")
    llm = LLM(
        model=model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="auto",
        gpu_memory_utilization=0.4,
        enable_lora=use_lora
    )
    
    lora_request = None
    if use_lora:
        lora_request = LoRARequest("degaml_policy_adapter", 1, lora_adapter_path)
    
    # --- Dynamic Generation Loop ---
    print(f"Starting generation process...")
    if family_filter:
        print(f"Single-family mode: Only accepting '{active_families[0]}' hypotheses.")
        print(f"Target: {num_generations} unique samples.")
    else:
        print(f"Constraints: 1. At least 2 samples per family. 2. Total unique samples >= {num_generations}.")
    print("Strategy: Shuffling prompt options every batch to avoid mode collapse.")
    
    unique_outputs = set()
    family_counts = {fam: 0 for fam in active_families}
    
    MAX_TOTAL_ATTEMPTS = 500  # Safety limit
    total_attempts = 0
    
    # Initialize progress bar
    pbar = tqdm(total=num_generations, desc="Generating Hypotheses")
    
    # Continue until constraints are satisfied
    def constraints_satisfied():
        if len(active_families) == 1:
            return len(unique_outputs) >= num_generations
        else:
            return min(family_counts.values()) >= 2 and len(unique_outputs) >= num_generations
    
    while not constraints_satisfied():
        if total_attempts >= MAX_TOTAL_ATTEMPTS:
            print(f"\n⚠️  Reached safety limit of {MAX_TOTAL_ATTEMPTS} attempts. Stopping early.")
            break
        
        # --- Dynamic Prompt Construction ---
        # Shuffle the descriptions to avoid positional bias
        current_blocks = option_blocks.copy()
        random.shuffle(current_blocks)
        
        # Re-assemble body with correct numbering
        dynamic_body = ""
        for idx, block in enumerate(current_blocks):
            dynamic_body += f"\n{idx+1}. {block}\n"
        
        full_prompt = PROMPT_HEADER + dynamic_body + PROMPT_FOOTER
        
        # Tokenize and format
        messages = [{"role": "user", "content": full_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Generate batch
        prompts = [formatted_prompt] * 5
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
        total_attempts += len(outputs)
        
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            try:
                start_index = generated_text.find('{')
                end_index = generated_text.rfind('}')
                
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    json_str = generated_text[start_index : end_index + 1]
                    data = json.loads(json_str)
                    
                    # --- Validation Logic ---
                    family = data.get("family")
                    
                    # 1. Check if family is recognized
                    if family not in family_counts:
                        continue
                    
                    # 2. TTT Constraint: Reject if ttl_steps > 50
                    if family == "TTT":
                        steps = data.get("ttl_steps", 0)
                        if steps > 50:
                            continue
                    
                    # --- Add to Collection ---
                    if json_str not in unique_outputs:
                        unique_outputs.add(json_str)
                        family_counts[family] += 1
                        
                        # Update progress bar (visual cap at num_generations)
                        if len(unique_outputs) <= num_generations:
                            pbar.update(1)
                        
                        # Update status display
                        pbar.set_postfix({
                            "Total": len(unique_outputs),
                            "MinFam": min(family_counts.values()),
                            **family_counts
                        })
            
            except (json.JSONDecodeError, ValueError):
                continue
    
    pbar.close()
    
    print(f"\nCollection Complete.")
    print(f"Final Counts: {family_counts}")
    print(f"Saving outputs to {output_file}...")
    
    final_outputs = list(unique_outputs)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, json_output in enumerate(final_outputs):
            f.write(f"--- Generation {i + 1} ---\n\n")
            f.write(json_output)
            f.write("\n\n" + "=" * 80 + "\n\n")
    
    print("\n" + "=" * 30 + " Process Complete " + "=" * 30)
    print(f"Successfully generated and saved {len(final_outputs)} unique responses.")
    print(f"File location: {output_file}")
    print("=" * 82)


if __name__ == "__main__":
    fire.Fire(generate_from_prompt)
