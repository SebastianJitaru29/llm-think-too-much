import pandas as pd
from pathlib import Path

base_df = pd.read_parquet(Path(__file__).parent / "TokenSkip-7B.parquet")
fine_tuned = pd.read_parquet(Path(__file__).parent / "1.5B_Epoch5.parquet")

print(base_df["length"].max())
avg_length_base = base_df["length"].mean()
num_correct_base = base_df["is_correct"].sum()

# For fine_tuned
avg_length_ft = fine_tuned["length"].mean()
num_correct_ft = fine_tuned["is_correct"].sum()

print("Base – avg length:", avg_length_base)
print("Base – # correct:", num_correct_base)

print("Fine-tuned – avg length:", avg_length_ft)
print("Fine-tuned – # correct:", num_correct_ft)