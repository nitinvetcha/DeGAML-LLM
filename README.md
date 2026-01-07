# DeGAML-LLM: Decoupled Generalization and Adaptation Meta-Learning for Large Language Models

<div align="center">

![DeGAML-LLM Architecture](assets/DeGAML-LLM-comparison-LLM.png)

**[Paper]** | **[Citation](#citation)** | **[Documentation](docs/)** | **[Checkpoints](docs/CHECKPOINTS.md)**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-spotlight-red.svg)]()

</div>

## 📋 Overview

**DeGAML-LLM** introduces a novel meta-learning framework that explicitly decouples generalization and adaptation for large language models, addressing fundamental limitations in existing approaches like MAML-en-LLM and ABMLL.

### Key Innovation

Traditional meta-learning for LLMs conflates two distinct objectives:
1. **Generalization**: Learning task-agnostic representations across task distributions
2. **Adaptation**: Enabling rapid task-specific refinement

DeGAML-LLM separates these through dedicated modules operating in distinct parameter spaces:

- **🔮 Generalization Module** ($\mathcal{G}_\phi$): Learns to generate LoRA adapter parameters from task prompts using a hyperconvolutional decoder trained on checkpoint trajectories
- **⚡ Adaptation Module** ($\pi_\psi$): Refines generated parameters via an RL policy that selects from four adaptation families (TTT, TTS, LoRA Mixing, Latent Space)

**Critical Design**: Gradients from adaptation *do not* flow back to the generalization module, ensuring true decoupling.

### Performance Highlights

✨ **State-of-the-art results** on common-sense reasoning, mathematics, logic, social, medical, and coding benchmarks  
🚀 **Outperforms** MAML-en-LLM, ABMLL, and standard multi-task baselines  
⚙️ **Flexible adaptation** via four distinct adaptation families with automatic strategy selection  
🎯 **Strong generalization** to out-of-domain tasks without task-specific fine-tuning

---

## 🏗️ Architecture

DeGAML-LLM consists of two key components trained sequentially:

![Architecture Diagram](assets/DeGAML-LLM-LLM.png)

### 1. Generalization Module (Parameter Generator)

- **Input**: Task prompts (unlabeled examples from test set)
- **Output**: Distribution over LoRA adapter parameters
- **Architecture**: 
  - Sentence-BERT encoder (all-MiniLM-L6-v2) for task embedding
  - Hyperconvolutional decoder for parameter generation
  - Parameter tokenization with 2D positional embeddings
- **Training**: Offline via MSE loss on collected LoRA checkpoints (no adaptation)

### 2. Adaptation Module (RL Policy)

- **Input**: Generated adapter parameters + validation performance
- **Output**: Adaptation strategy selection and refinement
- **Adaptation Families**:
  - **TTT (Test-Time Training)**: Fine-tune adapters on unlabeled test data via perplexity minimization
  - **TTS (Test-Time Scaling)**: Ensemble multiple adapters via max-confidence or majority vote
  - **LoRA Mixing**: Interpolate LoRA subspaces using two-subspace (TS) mixing
  - **Latent Space**: Optimize SLOT vectors (sample-specific latent parameters)
- **Training**: Online via ReST^EM with frozen generator (gradients detached)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DeGAML-LLM.git
cd DeGAML-LLM

# Create environment and install dependencies
conda create -n degaml python=3.12
conda activate degaml
pip install -r requirements.txt
```

### Environment Setup

Configure paths via environment variables (optional):

```bash
export DEGAML_DATA_ROOT="./data"
export DEGAML_OUTPUT_ROOT="./outputs"  
export DEGAML_CHECKPOINT_ROOT="./checkpoints"
export DEGAML_MODEL_ROOT="./models"
```

### Download Models

```bash
# Download base LLM (Qwen2.5-0.5B or 1.5B)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/Qwen2.5-0.5B-Instruct

# Download Sentence-BERT encoder
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 --local-dir ./models/all-MiniLM-L12-v2
```

### Basic Usage

#### 1. Baseline Evaluation (No Adaptation)

Generate adapters directly from task prompts:

```bash
python -m degaml.core.baseline \
    --eval_dataset ARC-c \
    --test_dataset ARC-c \
    --num_samples 25
```

#### 2. Generate Hypotheses

Use the RL policy to propose adaptation strategies:

```bash
python -m degaml.core.hypothesis_generation \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --lora_adapter_path ./checkpoints/policy_adapter \
    --num_generations 20 \
    --output_file ./outputs/hypotheses.txt
```

#### 3. Run Adaptation

Execute adaptation strategies (example with TTT):

```bash
python -m degaml.adaptation.test_time_training \
    --eval_dataset ARC-c \
    --test_dataset ARC-c \
    --ttl_steps 5 \
    --learning_rate 1e-5 \
    --batch_size 4
```

For complete pipeline examples, see [docs/USAGE.md](docs/USAGE.md).

---

## 📊 Experimental Results

### In-Domain Tasks (Common-Sense Reasoning)

| Method | ARC-c | HellaSwag | BoolQ | PIQA | WinoGrande | Avg |
|--------|-------|-----------|-------|------|------------|-----|
| Multi-Task Baseline | 45.2 | 52.1 | 67.3 | 72.4 | 58.9 | 59.2 |
| MAML-en-LLM | 47.8 | 54.6 | 69.1 | 73.8 | 60.2 | 61.1 |
| ABMLL | 48.3 | 55.2 | 70.4 | 74.1 | 61.3 | 61.9 |
| **DeGAML-LLM** | **51.7** | **58.4** | **73.2** | **76.9** | **64.1** | **64.9** |

### Out-of-Domain Tasks

| Method | GSM-8K | MATH | LogiQA | SocialIQA | MedQA | HumanEval |
|--------|--------|------|--------|-----------|-------|-----------|
| Multi-Task | 32.1 | 18.4 | 28.7 | 42.3 | 35.6 | 24.8 |
| MAML-en-LLM | 34.5 | 20.1 | 30.2 | 44.1 | 37.2 | 26.3 |
| **DeGAML-LLM** | **38.9** | **23.7** | **33.8** | **47.6** | **41.4** | **29.1** |

> **Note**: Results with Qwen2.5-0.5B-Instruct. See paper for complete results across model scales.

---

## 📚 Repository Structure

```
DeGAML-LLM/
├── degaml/                        # Main package
│   ├── core/                      # Core pipeline modules
│   │   ├── baseline.py            # Baseline adapter generation
│   │   ├── hypothesis_generation.py  # RL-based hypothesis generation
│   │   ├── accuracy.py            # Accuracy calculation
│   │   └── mega.py                # Pipeline orchestrator
│   ├── adaptation/                # Adaptation family modules
│   │   ├── test_time_training.py  # TTT implementation
│   │   ├── test_time_scaling.py   # TTS ensembling
│   │   ├── lora_mixing.py         # LoRA subspace mixing
│   │   └── latent_space.py        # SLOT vectors
│   ├── generator/                 # Parameter generator (from DnD)
│   │   ├── dataset/              # Dataset handling
│   │   ├── model/                # Generator model
│   │   ├── module/               # Hyperconvolution modules
│   │   ├── tokenizer/            # Parameter tokenization
│   │   └── tools/                # Utilities
│   ├── policy/                    # RL policy training
│   ├── utils/                     # Shared utilities
│   │   ├── paths.py              # Path configuration
│   │   └── config.py             # Config management
│   └── ablation/                  # Ablation study scripts
├── configs/                       # Configuration files
├── docs/                          # Documentation
├── scripts/                       # Training/inference scripts
├── assets/                        # Paper figures
└── requirements.txt               # Dependencies
```

---

## 🔧 Advanced Usage

### Running Ablation Studies

Isolate contributions of individual adaptation families:

```bash
python -m degaml.ablation.ablation_runner \
    --eval_dataset ARC-c \
    --test_dataset MATH-MC \
    --family TTT \
    --num_samples 25 \
    --iterations 1
```

### Training the Parameter Generator

See [Drag-and-Drop-LLMs](https://github.com/hiyouga/Drag-and-Drop-LLMs) for generator training instructions. Key steps:

1. Collect LoRA checkpoints across meta-training tasks
2. Calculate importance scores for parameter tokenization
3. Train hyperconvolutional decoder via MSE loss

### Training the RL Policy

```bash
python -m degaml.policy.train_policy \
    --meta_train_tasks "ARC-c,HellaSwag,BoolQ" \
    --num_iterations 10 \
    --reward_type accuracy_improvement
```

---

## 📖 Documentation

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed installation and setup instructions
- **[Usage Guide](docs/USAGE.md)**: Complete usage examples and tutorials
- **[Architecture](docs/ARCHITECTURE.md)**: In-depth architecture explanation
- **[Checkpoints](docs/CHECKPOINTS.md)**: Pre-trained model downloads

---

## 🎯 Citation

If you use DeGAML-LLM in your research, please cite our paper:

```bibtex
@inproceedings{degaml-llm2025,
  title={Decoupled Generalization and Adaptation Meta-Learning for Large Language Models},
  author={[Authors]},
  booktitle={[Conference]},
  year={2025}
}
```

---

## 🙏 Acknowledgments

- **Drag-and-Drop-LLMs** for the parameter generator architecture baseline
- **LLaMA-Factory** for training and inference infrastructure
- **vLLM** for efficient LLM inference
- **Sentence-Transformers** for text embedding models

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## 📧 Contact

For questions and feedback, please open an issue or contact [contact email].

---

<div align="center">

**Star ⭐ this repository if you find it helpful!**

Made with ❤️ for advancing meta-learning in LLMs

</div>
