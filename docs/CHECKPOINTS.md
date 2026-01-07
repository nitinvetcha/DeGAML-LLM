# Pre-trained Checkpoints for DeGAML-LLM

This document provides information about downloading and using pre-trained checkpoints.

> **Note**: Checkpoints will be made available upon paper publication. This page will be updated with HuggingFace links.

---

## Available Checkpoints

### Generalization Module

The parameter generator is trained offline on collected LoRA checkpoint trajectories.

| Model Size | Meta-Training Tasks | Checkpoint Size | Download Link |
|------------|---------------------|-----------------|---------------|
| Qwen2.5-0.5B | Common-sense reasoning (5 tasks) | ~2.5GB | Coming soon |
| Qwen2.5-1.5B | Common-sense reasoning (5 tasks) | ~5.8GB | Coming soon |
| Qwen2.5-0.5B | Multi-domain (12 tasks) | ~3.1GB | Coming soon |
| Qwen2.5-1.5B | Multi-domain (12 tasks) | ~7.2GB | Coming soon |

**Meta-Training Tasks**:
- Common-sense: ARC-c, HellaSwag, BoolQ, PIQA, WinoGrande
- Multi-domain: + GSM-8K, MATH, LogiQA, SocialIQA, MedQA, HumanEval, CodeMMLU

### Adaptation Module

The RL policy is trained online with the generator frozen.

| Model Size | Training Iterations | Checkpoint Size | Download Link |
|------------|---------------------|-----------------|---------------|
| Qwen2.5-0.5B | 10 iterations | ~150MB | Coming soon |
| Qwen2.5-1.5B | 10 iterations | ~450MB | Coming soon |

### Collected LoRA Trajectories

Pre-collected checkpoint trajectories for training your own generator:

| Dataset Collection | Tasks | Total Checkpoints | Archive Size | Download Link |
|-------------------|-------|-------------------|--------------|---------------|
| Common-sense | 5 tasks × 125 ckpts | 625 | ~15GB | Coming soon |
| Mathematics | 2 tasks × 125 ckpts | 250 | ~6GB | Coming soon |
| Full meta-training | 12 tasks × 125 ckpts | 1,500 | ~38GB | Coming soon |

---

## Download Instructions

### Automatic Download (Recommended)

```bash
# Download parameter generator
python scripts/download_checkpoints.py \
    --model_size 0.5B \
    --checkpoint_type generator \
    --meta_training common_sense

# Download policy adapter
python scripts/download_checkpoints.py \
    --model_size 0.5B \
    --checkpoint_type policy
```

### Manual Download

1. Visit the HuggingFace model page: `[Link TBD]`
2. Download desired checkpoint files
3. Place in `$DEGAML_CHECKPOINT_ROOT/` with appropriate naming:
   - Generator: `qwen0.5lora__ARC-c4000.pth`
   - Policy: `policy_adapter_iter10/`

---

## Using Checkpoints

### With Parameter Generator

```bash
# Baseline generation using pre-trained generator
python -m degaml.core.baseline \
    --eval_dataset ARC-c \
    --test_dataset ARC-c \
    --checkpoint_path $DEGAML_CHECKPOINT_ROOT/qwen0.5lora__ARC-c4000.pth
```

### With Policy Adapter

```bash
# Hypothesis generation with policy adapter
python -m degaml.core.hypothesis_generation \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --lora_adapter_path $DEGAML_CHECKPOINT_ROOT/policy_adapter_iter10 \
    --num_generations 20
```

---

## Checkpoint Directory Structure

Organize checkpoints as follows:

```
checkpoints/
├── generators/
│   ├── qwen0.5lora__ARC-c4000.pth
│   ├── qwen1.5lora__ARC-c4000.pth
│   └── ...
├── policy_adapters/
│   ├── policy_adapter_iter10/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── ...
└── lora_trajectories/
    ├── ARC-c/
    │   ├── checkpoint_0.pth
    │   ├── checkpoint_1.pth
    │   └── ...
    └── ...
```

---

## Training Your Own Checkpoints

### Parameter Generator

The parameter generator is trained on collected LoRA checkpoint trajectories using the hyperconvolutional decoder architecture included in this repository.

Key steps:
1. Collect LoRA checkpoints across tasks
2. Calculate importance scores
3. Train hyperconvolutional decoder

```bash
# Example training command (self-contained in this repository)
# Training scripts will be added in future releases
# For now, pre-trained checkpoints are provided above
```

### Policy Adapter

```bash
python -m degaml.policy.train_policy \
    --meta_train_tasks "ARC-c,HellaSwag,BoolQ,PIQA,WinoGrande" \
    --base_model_path $DEGAML_MODEL_ROOT/Qwen2.5-0.5B-Instruct \
    --generator_checkpoint $DEGAML_CHECKPOINT_ROOT/qwen0.5lora__ARC-c4000.pth \
    --num_iterations 10 \
    --output_dir $DEGAML_CHECKPOINT_ROOT/policy_adapters
```

---

## Verification

Verify checkpoint integrity:

```bash
# Check generator checkpoint
python -c "
import torch
ckpt = torch.load('checkpoints/qwen0.5lora__ARC-c4000.pth', map_location='cpu')
print(f'✅ Generator checkpoint loaded successfully')
print(f'Keys: {list(ckpt.keys())}')
"

# Check policy adapter
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
model = PeftModel.from_pretrained(model, 'checkpoints/policy_adapters/policy_adapter_iter10')
print('✅ Policy adapter loaded successfully')
"
```

---

## FAQ

**Q: Can I use checkpoints trained on different model sizes?**  
A: No, checkpoints are model-size specific. Use 0.5B checkpoints with 0.5B models, etc.

**Q: How were these checkpoints trained?**  
A: See our paper Section 4 (Appendix: Training Procedure) for full training details.

**Q: Can I fine-tune the checkpoints?**  
A: Yes! You can continue training the generator or policy adapter on additional tasks.

**Q: What license do checkpoints have?**  
A: Same as code - Apache 2.0. See LICENSE file.

---

## Updates

This page will be updated with:
- [ ] HuggingFace checkpoint links (upon publication)
- [ ] Additional checkpoint variants
- [ ] Fine-tuned checkpoints on specialized domains
- [ ] Community-contributed checkpoints

Check back regularly or watch the repository for updates!
