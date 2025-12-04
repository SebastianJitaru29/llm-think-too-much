import pandas as pd
from pathlib import Path

# Path to your parquet file
parquet_file = Path(__file__).parent / "generated_math" / "generated_part1.parquet"

# Read the parquet file
df = pd.read_parquet(parquet_file)

# Extract and rename relevant columns
df_extracted = pd.DataFrame({
    "question_id": df["question_id"],
    "prompt": df["prompt"],
    "solution_col": df["solution_col"],
    "generated_think_text": df["generated_think_text"],
    "generated_text": df["generated_text"],
    "target_think_tokens": df["target_think_tokens"].astype(int),
    "generated_think_tokens": df["generated_think_tokens"].astype(int),
    "is_correct": df["is_correct"].astype(bool)
})

# Iterate over rows and print length of generated_text, target_think_tokens, and is_correct
for idx, row in df_extracted.iterrows():
    generated_text_len = len(str(row["generated_text"]).split(" "))
    print(f"Generated Text Length: {generated_text_len}, "
          f"Target Think Tokens: {row['target_think_tokens']}, "
          f"Is Correct: {row['is_correct']}")
