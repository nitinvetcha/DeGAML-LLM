# Installation Guide for DeGAML-LLM

This guide provides detailed installation instructions for DeGAML-LLM.

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with at least 24GB VRAM (RTX 3090/4090, A5000, A6000, or better)
  - For larger models (1.5B+): 48GB+ VRAM recommended  
- **RAM**: 64GB+ system RAM recommended
- **Storage**: 100GB+ free space for models and checkpoints

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **CUDA**: 12.1 or higher
- **Python**: 3.12+
- **conda** or **venv** for environment management

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DeGAML-LLM.git
cd DeGAML-LLM
```

### 2. Create Python Environment

#### Option A: Using conda (Recommended)

```bash
# Create environment
conda create -n degaml python=3.12
conda activate degaml

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### Option B: Using venv

```bash
python3.12 -m venv venv
source venv/bin/activate

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- `transformers`, `accelerate`, `peft` for LLM infrastructure
- `vllm` for efficient inference
- `llamafactory` for training utilities
- DeGAML-specific dependencies

### 4. Set Up Directory Structure

```bash
mkdir -p data models checkpoints outputs
```

### 5. Configure Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export DEGAML_DATA_ROOT="/path/to/DeGAML-LLM/data"
export DEGAML_OUTPUT_ROOT="/path/to/DeGAML-LLM/outputs"
export DEGAML_CHECKPOINT_ROOT="/path/to/DeGAML-LLM/checkpoints"
export DEGAML_MODEL_ROOT="/path/to/DeGAML-LLM/models"
```

Then reload:
```bash
source ~/.bashrc
```

### 6. Download Base Models

#### Qwen Models

```bash
# Qwen2.5-0.5B-Instruct (smaller, faster)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir $DEGAML_MODEL_ROOT/Qwen2.5-0.5B-Instruct

# Qwen2.5-1.5B-Instruct (larger, better performance)
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir $DEGAML_MODEL_ROOT/Qwen2.5-1.5B-Instruct
```

#### Sentence-BERT Encoder

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 \
    --local-dir $DEGAML_MODEL_ROOT/all-MiniLM-L12-v2
```

### 7. Download Datasets

Follow instructions in `docs/DATASETS.md` to set up evaluation benchmarks:
- Common-sense reasoning: ARC-c, ARC-e, HellaSwag, BoolQ, PIQA, WinoGrande  
- Mathematics: GSM-8K, MATH
- Logic: LogiQA
- Social: SocialIQA
- Medical: MedQA
- Coding: HumanEval

### 8. (Optional) Download Pre-trained Checkpoints

See `docs/CHECKPOINTS.md` for links to pre-trained:
- Parameter generator checkpoints
- Policy adapter checkpoints
- Collected LoRA checkpoint trajectories

---

## Verification

### Test Installation

```bash
# Test imports
python -c "import degaml; from degaml.core import accuracy; print('✅ DeGAML-LLM installed successfully!')"

# Test CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Test vLLM
python -c "from vllm import LLM; print('✅ vLLM installed successfully!')"
```

### Run Quick Test

```bash
# Run accuracy calculation on a sample file
python -m degaml.core.accuracy --help
```

---

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:
1. Reduce batch size in configs
2. Use smaller model (0.5B instead of 1.5B)
3. Adjust `gpu_memory_utilization` parameter in vLLM

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Ensure DeGAML-LLM is in PYTHONPATH
export PYTHONPATH="/path/to/DeGAML-LLM:$PYTHONPATH"
```

### vLLM Installation Issues

vLLM requires specific CUDA versions. If installation fails:
```bash
# Check CUDA version
nvcc --version

# Install matching vLLM version
pip install vllm==0.3.0  # Adjust version as needed
```

### LLaMA-Factory Installation

If LLaMA-Factory installation fails:
```bash
# Clone and install manually
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout v0.9.2
pip install -e ".[torch,metrics]"
cd ..
```

---

## Next Steps

After successful installation:
1. Read [USAGE.md](USAGE.md) for usage examples
2. Download datasets (see `docs/DATASETS.md`)
3. Download or train checkpoints (see `docs/CHECKPOINTS.md`)
4. Run baseline evaluation
5. Explore adaptation strategies

---

## Docker Installation (Alternative)

For a containerized setup:

```bash
# Build Docker image
docker build -t degaml-llm .

# Run container with GPU support
docker run --gpus all -it degaml-llm
```

> **Note**: Dockerfile coming soon!

---

## Support

For installation issues, please:
1. Check existing [GitHub Issues](https://github.com/YOUR_USERNAME/DeGAML-LLM/issues)
2. Open a new issue with:
   - OS and hardware details
   - Python and CUDA versions
   - Full error message
   - Installation method used
