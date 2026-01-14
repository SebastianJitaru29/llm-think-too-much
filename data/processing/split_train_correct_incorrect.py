import pandas as pd
from pathlib import Path
import numpy as np

df = pd.read_parquet(Path(__file__).parent / "train.parquet")


levels = [f"Level {i}" for i in range(1, 6)] + ["aimee"]
df["category"] = df["level"].apply(lambda x: x if x in levels else "something else")


df["is_correct_bool"] = df["is_correct"].astype(str).str.strip().str.lower().map({"true": True, "false": False})

# group by question_id and category, get per-question correctness
per_question = (
    df.groupby(["question_id", "category"], observed=True)["is_correct_bool"]
      .any()
      .reset_index()
)

counts = per_question.groupby("category")["is_correct_bool"].value_counts().unstack(fill_value=0)

print(counts)

correct_questions = per_question[per_question["is_correct_bool"] == True]
incorrect_questions = per_question[per_question["is_correct_bool"] == False]

frac_correct = 0.6  
frac_incorrect = 0.4  

sampled_correct = correct_questions.sample(n=int(len(incorrect_questions)* (frac_correct/frac_incorrect)) , random_state=42)
sampled_incorrect = incorrect_questions

split_df = pd.concat([sampled_correct, sampled_incorrect]).sample(frac=1, random_state=42).reset_index(drop=True)

selected_question_ids = split_df["question_id"].unique()

filtered_rows_df = df[df["question_id"].isin(selected_question_ids)].copy()


filtered_rows_df.to_parquet("split_train_c60_inc40.parquet", index=False)

print(filtered_rows_df.head())
print("Total rows in new DataFrame:", len(filtered_rows_df))

