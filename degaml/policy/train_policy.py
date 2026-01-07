# train_policy.py (Corrected Version)
# Rest^EM adaptation strategy training script using LoRA and Hugging Face Trainer.

import json
import argparse
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a policy model using ReST^EM with LoRA.")
    parser.add_argument("--actions_file", type=str, required=True, help="Path to the JSON file with prompts and actions (responses).")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the JSON file with binary rewards.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Name or path of the base model to be trained.")
    parser.add_argument("--output_dir", type=str, default="./restem_trained_model", help="Directory to save the trained model adapter.")
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank for LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha parameter for LoRA.")
    parser.add_argument("--num_train_epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    return parser.parse_args()

def main():
    args = parse_args()

    # --- 1. Load and Filter Data ---
    print("Step 1: Loading and filtering data for successful actions...")
    with open(args.actions_file, 'r') as f:
        actions_data = json.load(f)
    with open(args.results_file, 'r') as f:
        results_data = json.load(f)

    successful_interactions = []
    for sample_idx_str, result_info in results_data.items():
        if result_info['correct']['0']: # Check for reward == 1
            task_id = result_info['task_id']['0']
            interaction = actions_data[task_id][sample_idx_str]
            # **FIX:** Instead of concatenating strings, create a structured message list
            # This allows the tokenizer to apply the correct chat template.
            successful_interactions.append({
                "messages": [
                    {"role": "user", "content": interaction['prompt']},
                    {"role": "assistant", "content": interaction['response']}
                ]
            })

    print(f"Found {len(successful_interactions)} successful interactions to train on.")
    if not successful_interactions:
        print("No successful interactions found. Exiting.")
        return

    # --- 2. Load Model and Tokenizer ---
    print(f"Step 2: Loading base model and tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
        # **FIX:** Removed `device_map="auto"`. DDP will handle device placement.
    )

    # --- 3. Configure LoRA ---
    print("Step 3: Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Create and Tokenize Dataset ---
    print("Step 4: Creating and tokenizing the dataset...")
    dataset = Dataset.from_list(successful_interactions)

    def tokenize_function(examples):
        # **FIX:** This function is now much more robust. It finds the special tokens
        # that mark the beginning of the assistant's turn and masks everything before it.
        full_prompts = [tokenizer.apply_chat_template(ex, tokenize=False) for ex in examples["messages"]]
        tokenized = tokenizer(full_prompts, truncation=True, max_length=4096, padding="longest")
        
        labels = [row[:] for row in tokenized["input_ids"]]
        
        # This sequence corresponds to `<|im_start|>assistant\n` for Qwen2
        assistant_token_sequence = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        seq_len = len(assistant_token_sequence)

        for i, label_row in enumerate(labels):
            # Find the sequence of tokens marking the assistant's response
            for j in range(len(label_row) - seq_len):
                if label_row[j:j+seq_len] == assistant_token_sequence:
                    # Mask all tokens up to and including this sequence
                    for k in range(j + seq_len):
                        labels[i][k] = -100
                    break
        
        tokenized["labels"] = labels
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["messages"])

    # --- 5. Train the Model ---
    print("Step 5: Setting up and starting the training process...")
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=True,
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---")

    # --- 6. Save the Final Model ---
    # Only the main process (rank 0) should save the model
    if trainer.is_world_process_zero():
        print(f"Step 6: Saving the final trained LoRA adapter to {args.output_dir}")
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        print("\n" + "=" * 25 + " Process Complete " + "=" * 25)
        print(f"Successfully trained the policy. The LoRA adapter is saved in: {args.output_dir}")
        print("You can now use this adapter with the base model for the next iteration of generation.")
        print("=" * 72)


if __name__ == "__main__":
    main()