# CoT Optimization via Adaptive Token Budgets

Optimizing Chain-of-Thought reasoning by learning to predict the minimum token budget required for correct problem solving, reducing computational cost while maintaining accuracy.

## Overview

Large language models benefit from extended reasoning (Chain-of-Thought), but longer reasoning increases latency and cost. This project explores and compares two approaches to optimize the trade-off between reasoning length and computational efficiency.

**Key Insight:** Different problems require different amounts of reasoning. Simple problems may need only 100 tokens, while complex ones might require 2000+. The challenge is learning when to allocate more or less compute.

### Two Optimization Strategies

**1. Neural Regressor Approach**
- Trains a lightweight MLP (1.5M parameters) to predict optimal token budgets from problem representations
- Keeps the base LLM frozen and uses an external predictor
- Supports two inference modes:
  - **Static**: One-shot prediction before generation
  - **Dynamic**: Adaptive re-prediction every 50 tokens during generation
- **Advantages**: Modular, lightweight, can swap base models without retraining

**2. Direct Preference Optimization (DPO)**
- Fine-tunes the base LLM using preference pairs (short correct vs long incorrect solutions)
- Teaches the model to inherently generate efficient reasoning
- Uses LoRA adapters for parameter-efficient fine-tuning
- **Advantages**: No external predictor needed, model learns efficiency directly

### Methodology

1. **Data Generation**: Generate solutions across 10 token budgets (100-2500) for thousands of math problems
2. **Feature Extraction**: Extract hidden states and track correctness at each budget level
3. **Training**:
   - Regressor: Train MLP on initial hidden states → budget predictions
   - DPO: Fine-tune LLM on preference pairs (min-token correct > max-token incorrect)
4. **Evaluation**: Compare both approaches on:
   - Token efficiency (average tokens used)
   - Accuracy preservation
   - Training cost and inference overhead
   - Generalization across problem difficulty levels

## Setup

### 1. Install Dependencies

Create and activate a virtual environment, then install requirements. We added torch with cuda-12.8 enabled into the requirements, if your system requires a different version, please change it in the requirements.txt:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
bash download_dataset.sh
```

Downloads:
- MATH dataset (Competition Math problems)
- AIME dataset (1983-2024)

### 3. Download Models

```bash
bash download_models.sh
```

Downloads:
- `L1-Qwen-1.5B-Exact` (fine-tuned reasoning model)
- `DeepSeek-R1-Distill-Qwen-1.5B` (baseline)

## Usage

### Generate Training Data

**For MATH dataset:**
```bash
python launch_experiments.py \
  --data ./data/math.parquet \
  --model-path ./models/L1-Qwen-1.5B-Exact \
  --generated-dir ./generated_math \
  --hidden-dir ./hidden_math \
  --batch-size 4
```

**For AIME dataset:**
```bash
python launch_aime_experiments.py \
  --data ./data/aime.parquet \
  --model-path ./models/L1-Qwen-1.5B-Exact \
  --generated-dir ./generated_aime \
  --hidden-dir ./hidden_aime \
  --batch-size 4
```

Generates solutions across 10 token budgets (100-2500) and extracts hidden states at 50-token intervals.

### Train Regressor

Train the token predictor on generated hidden states:

```bash
cd regressor
python train.py
```

Trains a 4-layer MLP (1536→256→256→256→10) with dropout to predict correctness across 10 token budgets.

### Run Experiments

**Static Regressor (single prediction):**
```bash
python launch_regressor.py --type static --batch 8
```

**Dynamic Regressor (adaptive):**
```bash
python launch_regressor.py --type dynamic --batch 4 --every 50
```

Results saved to `static_regressor_results/` or `dynamic_regressor_results/`.

### DPO Fine-Tuning

Train an alternative approach using Direct Preference Optimization. This method fine-tunes the LLM to inherently generate shorter, more efficient reasoning:

```bash
cd fine_tuning_dpo
python fine_tuning.py
```

**How it works:**
- Creates preference pairs: chosen = minimum-token correct solution, rejected = maximum-token incorrect/inefficient solution
- Uses LoRA (Low-Rank Adaptation) to efficiently fine-tune the model
- Trains for 20 epochs with batch size 8 (via gradient accumulation)
- Cleans generated text (removes duplicate answers, special token artifacts)
- Filters out Chinese responses and empty generations

Requires `train.parquet` with generated solutions from `launch_experiments.py`. Outputs LoRA adapters to `./qwen-1.5B-dpo-lora/`.

This serves as a baseline to compare against the regressor approach for token efficiency optimization.

## Project Structure

```
.
├── regressor/              # Regressor architecture & training
│   ├── architecture.py     # JAX/Flax MLP definition
│   ├── train.py           # Training script
│   └── regressor.pkl      # Trained model weights
├── launch_experiments.py   # Generate data for MATH dataset
├── launch_aime_experiments.py  # Generate data for AIME
├── launch_regressor.py    # Run static/dynamic inference
├── fine_tuning_dpo/       # DPO training code
├── download_dataset.sh    # Dataset download script
└── download_models.sh     # Model download script
```


