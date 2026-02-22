#!/usr/bin/env bash
mkdir -p data

python3 - <<'EOF'
import pandas as pd
from pathlib import Path

# Load parquet from HF dataset and save locally
df = pd.read_parquet("hf://datasets/qwedsacf/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet")
df.to_parquet("./data/math.parquet")

# Load CSV from another HF dataset and overwrite (or append) locally
df = pd.read_csv("hf://datasets/di-zhang-fdu/AIME_1983_2024/AIME_Dataset_1983_2024.csv")
df.to_parquet("./data/aime.parquet")
EOF
