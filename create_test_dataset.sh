mkdir -p data

python3 - <<'EOF'
import pandas as pd
from pathlib import Path

# Load parquet from HF dataset and save locally
df_500 = pd.read_json(
        "hf://datasets/HuggingFaceH4/MATH-500/test.jsonl",
        lines=True
)
df_500.to_parquet("./data/test_math.parquet")
EOF

