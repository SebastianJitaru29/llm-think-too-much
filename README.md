# CoT Optimization via Adaptive Token Budgets

Chain-of-Thought optimization using a learned regressor to predict optimal reasoning token budgets for mathematical problem solving.

## Overview

This project trains a neural regressor to predict the minimum token budget needed for an LLM to correctly solve math problems. Two inference modes are supported:
- **Static**: One-shot prediction of optimal token budget
- **Dynamic**: Adaptive prediction that updates during generation

## Setup

### 1. Install Dependencies

Create and activate a virtual environment, then install requirements:

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

### Train Regressor

Train the token predictor on generated hidden states:

```bash
cd regressor
python train.py
```

Trains a 4-layer MLP (1536→256→256→256→10) with dropout to predict correctness across 10 token budgets.

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

Train with Direct Preference Optimization using min-token correct vs max-token incorrect pairs:

```bash
cd fine_tuning_dpo
python fine_tuning.py
```

Requires `train.parquet` with generated solutions. Outputs LoRA adapters to `./qwen-1.5B-dpo-lora/`.

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

## Key Features

- **Multi-budget generation**: Tests 10 token budgets per problem (100-2500 tokens)
- **Hidden state extraction**: Captures model representations at regular intervals
- **Binary classification**: Learns which budgets produce correct answers
- **Adaptive inference**: Dynamic mode updates predictions during generation
- **DPO optimization**: Preference learning from token-efficiency pairs

