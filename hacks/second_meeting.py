import pandas as pd

# Train
df_train = pd.read_parquet("../data/train.parquet")
df_train = df_train.drop_duplicates("question_id")
df_train = df_train.drop(
    columns=["generated_think_text", "generated_text", "latency_sec"]
).rename(
    columns={"question_id": "id", "prompt": "problem", "solution_col": "solution"}
)

df_train.to_parquet("../data/actual_train.parquet", index=False)

# Test
df_test = pd.read_parquet("../data/test.parquet")
df_test = df_test.drop_duplicates("question_id")

df_test_aligned = df_test.drop(
    columns=["generated_think_text", "generated_text", "latency_sec", "is_correct", "target_think_tokens", "generated_think_tokens"]
).rename(
    columns={"question_id": "id", "prompt": "problem", "solution_col": "solution"}
)

e = pd.read_parquet("../data/eval_data.parquet")
e = e[e["dataset"] != "math-500"]
e["problem"] = e["problem"] + " Let’s think step by step inside and output the final answer within boxed{{}}."


df_test = pd.concat((df_test, e), ignore_index=True)
df_test.to_parquet("../data/actual_test.parquet", index=False)


