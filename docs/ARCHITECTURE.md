# DeGAML-LLM Architecture

This document provides an in-depth explanation of the DeGAML-LLM architecture and its components.

##  Overview

DeGAML-LLM consists of two sequential training stages with distinct modules:

1. **Stage 1 (Offline)**: Train Parameter Generator $\mathcal{G}_\phi$
2. **Stage 2 (Online)**: Train RL Policy $\pi_\psi$ with frozen generator

**Key Principle**: Gradients from Stage 2 do NOT flow back to $\mathcal{G}_\phi$ (detached computation graph).

---

## 1. Generalization Module ($\mathcal{G}_\phi$)

### Purpose
Learn task-agnostic representations by generating a distribution over LoRA adapter parameters conditioned on task prompts.

### Architecture Components

#### 1.1 Task Embedding Extraction

**Input**: Unlabeled task prompts (128 samples from test set)

**Encoder**: Sentence-BERT (all-MiniLM-L6-v2)
- Dimension: `[B, N, L, C]` where:
  - `B` = batch size
  - `N` = number of prompts (128)
  - `L` = sequence length (512)
  - `C` = hidden dimension (384)

**Output**: Task embedding $c_i \in \mathbb{R}^{128 \times 512 \times 384}$

#### 1.2 Parameter Tokenization

Transforms LoRA adapter weights into uniform tokens:

**Step 1**: Layer-wise splitting & normalization
```
W → [w[1], ..., w[I]] → [ŵ[1], ..., ŵ[I]]
```

**Step 2**: Uniform tokenization  
- Token size: `k` (e.g., 1024 for rank-8, 0.5B model)
- For Qwen2.5-0.5B (rank=8): 7 tokens of `8×128` per layer, padded to `10×130`
- For Qwen2.5-1.5B (rank=16): 6 tokens of `16×256` per layer, padded to `18×258`

**Step 3**: Positional embeddings
- 2D sinusoidal position embeddings
- First dimension: layer index
- Second dimension: in-layer token position

#### 1.3 Hyperconvolutional Decoder

Multi-layer 2D convolutions that generate adapter parameters:

**Architecture**:
```python
features = [
    (128, 384, 384),  # Input dimension (B, L, C)
    (128, 200, 300),  # Intermediate layers
    (128, 100, 256),
    (256, 50, 200),
    (512, 50, 200),
    (1024, 25, 200),
    (1024, 10, 200),
    (2048, 10, 200),
    (4296, 10, 130),  # Output dimension
]
```

**Convolution Types**:
1. **Width Conv** ($\text{Conv}_W$): Operates on `(C, L)` dimensions
2. **Height Conv** ($\text{Conv}_H$): Operates on `(B, L)` dimensions  
3. **Mixed Conv**: Combines both

**Kernel Size**: 9×9

**Output**: Generated LoRA parameters $\theta^a \sim q_\phi(\cdot | D^{tr})$

### Training Objective

**Loss**: Mean Squared Error (MSE) between generated and collected checkpoint parameters

$$\min_\phi \mathbb{E}_{(p_k, m_j) \sim \mathcal{D}_{\text{meta}}} \|\mathcal{G}_\phi(p_k) - m_j\|^2$$

where:
- $p_k$: prompt batch embeddings
- $m_j$: checkpoint parameters from LoRA training trajectories

**Key**: No adaptation mechanism involved - purely learns parameter manifold.

### Checkpoint Collection

For each meta-training task $\mathcal{T}_i$:
1. Pre-train LoRA adapters for 75 steps (lr=1e-4, bs=32)
2. Fine-tune for 50 steps (lr=1e-5)
3. Save checkpoint at each step
4. Total: ~125 checkpoints per task

Random pairing: ~5,000 (prompt, checkpoint) pairs per task.

---

## 2. Adaptation Module ($\pi_\psi$)

### Purpose
Refine generated adapter parameters via strategic adaptation, learning which strategy works best for each task.

### Adaptation Families

#### 2.1 Test-Time Training (TTT)

**Concept**: Fine-tune adapter on unlabeled test data via self-supervision

**Method**: Minimize input perplexity
$$\min_{\theta^a} \mathbb{E}_{x \sim D^{test}} [-\log p_{\theta^a}(x)]$$

**Hyperparameters**:
- `ttl_steps`: Number of optimization steps (1-50)
- `learning_rate`: Learning rate (1e-6 to 1e-4)
- `batch_size`: Batch size for TTL (2-8)
- `shuffle_data`: Whether to shuffle test data

**Implementation**: `degaml/adaptation/test_time_training.py`

#### 2.2 Test-Time Scaling (TTS)

**Concept**: Generate multiple adapters and ensemble predictions

**Methods**:
1. **Max Confidence**: Select prediction with highest probability
2. **Majority Vote**: Most frequent prediction across adapters

**Hyperparameters**:
- `num_lora_adpt`: Number of adapters to generate (2-10)
- `method`: "max_confidence" or "majority_vote"

**Implementation**: `degaml/adaptation/test_time_scaling.py`

#### 2.3 LoRA Mixing

**Concept**: Interpolate between LoRA subspaces using Two-Subspace (TS) mixing

**Formula**: $(A_1 + A_2)(B_1 + B_2)$ instead of $A_1B_1 + A_2B_2$

**Hyperparameters**:
- `lambda`: Mixing ratio (0.0 to 1.0)

**Implementation**: `degaml/adaptation/lora_mixing.py`

#### 2.4 Latent Space (SLOT)

**Concept**: Optimize sample-specific latent vectors added to hidden layer features

**Method**: Minimize cross-entropy on input prompt itself
$$\min_{\delta} \mathcal{L}_{CE}(f_{\theta}(x) + \delta, y)$$

**Hyperparameters**:
- `slot_steps`: Optimization steps per sample (1-5)
- `slot_lr`: Learning rate for SLOT (1e-3 to 1e-1)

**Implementation**: `degaml/adaptation/latent_space.py`

### RL Policy Training

**Algorithm**: ReST^EM (Reinforcement Learning with Self-Training via Expectation-Maximization)

**State** $s_t$: Current performance metrics (accuracy, confidence, etc.)

**Action** $a_t$: Adaptation strategy selection + hyperparameter perturbation
$$\theta_{t+1}^a = \theta_t^a + a_t, \quad a_t \sim \pi_\psi(a_t | s_t)$$

**Reward** $r_t$: Binary reward based on validation performance improvement
$$r_t = -\mathcal{L}_{\mathcal{T}_*}(\theta_{t+1}^a, D_*^{val})$$

**Key**: Generator outputs are **detached** - no gradients flow back to $\mathcal{G}_\phi$.

---

## 3. Pipeline Orchestration

The full pipeline is coordinated by `degaml/core/mega.py`:

### Execution Flow

```
Step 0: Baseline
├── Run BL.py → Generate adapters
└── Run AC.py → Calculate baseline accuracy

Step 1: Hypothesis Generation  
└── Run HG.py → Generate N adaptation hypotheses

Step 2: Hypothesis Evaluation
├── For each hypothesis:
│   ├── Dispatch to family script (TL/TS/LM/LS)
│   ├── Run adaptation
│   ├── Calculate accuracy (AC.py or AC_LS.py)
│   └── Assign reward
└── Select best hypothesis

Step 3: Policy Training
├── Convert experiences to PT format
├── Run PT.py → Update policy adapter
└── Save new adapter

Step 4: Iteration
└── Feed updated adapter to HG.py → Repeat from Step 1
```

### Checkpoint Chaining

**Between iterations**, best checkpoints carry forward:
- **TTT**: Best TTL-trained adapter
- **TTS**: Multiple generated adapters (ensemble list)
- **LoRA**: Lambda-mixed adapter
- **Latent**: Best SLOT vectors (saved separately)

**Implementation**: `base_checkpoint` dict in `mega.py`

---

## 4. Key Design Decisions

### Why Decouple?

**Problem with coupled approaches** (MAML, ABMLL):
- Single parameter space tries to optimize both:
  1. Cross-task generalization
  2. Fast adaptation dynamics
- Leads to suboptimal solutions for both objectives

**DeGAML-LLM solution**:
- $\mathcal{G}_\phi$: Learns WHAT good initializations look like
- $\pi_\psi$: Learns HOW to adapt them effectively
- No interference between objectives

### Why RL for Adaptation?

Traditional meta-learning:
- Update adaptation via gradient descent
- Limited to gradient-based strategies

DeGAML-LLM:
- RL policy can select from diverse non-differentiable strategies
- Mix gradient-based (TTT) and non-gradient (TTS, LoRA mixing)
- Learn task-specific strategy selection

### Why Detached Gradients?

If gradients flowed $\pi_\psi \rightarrow \mathcal{G}_\phi$:
- Generator would learn to produce "easy-to-adapt" parameters
- Not necessarily "good for the task" parameters
- Defeats purpose of decoupling

With detached gradients:
- Generator optimizes purely for task performance
- Policy adapts to generator's outputs independently

---

## 5. Computational Efficiency

### Generator Forward Pass
- **Input**: 128 prompts × 512 tokens
- **Compute**: Single forward through hyperconv decoder
- **Time**: ~100ms on A6000 GPU
- **Memory**: ~2GB VRAM

### Adaptation Strategies
| Strategy | Time per Task | Memory |
|----------|--------------|---------|
| Baseline | ~3 min | 6GB |
| TTT (5 steps) | ~5 min | 8GB |
| TTS (3 adapters) | ~9 min | 10GB |
| LoRA Mixing | ~3 min | 6GB |
| Latent (SLOT) | ~7 min | 7GB |

### Full Pipeline
- **1 iteration**: ~30-45 minutes (depends on num_hypotheses)
- **10 iterations**: ~6-8 hours on 4× A6000 GPUs

---

## 6. Code Organization Rationale

```
degaml/
├── core/           # Pipeline orchestration & core logic
├── adaptation/     # Swappable adaptation strategies
├── generator/      # Reusable parameter generation
├── policy/         # RL training (separate concern)
└── utils/          # Shared infrastructure
```

**Benefits**:
- **Modularity**: Easy to add new adaptation families
- **Testability**: Each component is independently testable  
- **Reusability**: Generator can be used without adaptation
- **Clarity**: Clear separation of concerns

---

For implementation details, see the source code and inline documentation.
