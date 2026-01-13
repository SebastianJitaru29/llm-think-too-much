mkdir -p data

python3 - <<'EOF'
import pandas as pd
from pathlib import Path
import numpy as np

# Load parquet from HF dataset and save locally
df_500 = pd.read_json(
        "hf://datasets/HuggingFaceH4/MATH-500/test.jsonl",
        lines=True
)
df_500.to_parquet("./data/test_math.parquet", index=False)

df_aime_ids = pd.read_parquet("./data/test_aime.parquet")["question_id"].unique()
df_aime = pd.read_parquet("./data/aime.parquet")
df_aime["question_id"] = np.arange(df_aime.shape[0])
df_aime = df_aime[df_aime["question_id"].isin(df_aime_ids)]
df_aime = df_aime.rename(columns={"Question": "problem", "Answer": "solution"})

df_all_problem = pd.concat((df_aime[["problem", "solution"]], df_500[["problem", "solution"]]), ignore_index=True)
df_all_problem.to_parquet("./data/test_all.parquet", index=False)
EOF

