import pandas as pd 
import os
import csv
from pathlib import Path
import re
from datasets import load_dataset

MATH1_DATASET_PATH = Path(__file__).parent.parent / "data" / "math1_results_long.parquet"
MATH2_DATASET_PATH = Path(__file__).parent.parent / "data" / "math2_results_long.parquet" #/"/home3/s3799042/data_nlp/MATH"
AIME_DATASET_PATH = Path(__file__).parent.parent / "data" / "aime_results_long_with_aime_id.parquet" #/home3/s3799042/data_nlp/AIMEE"

KEY = 42

TEST_SIZE_AIME = 250

def split_math500(generated_df, math500_df):
    match_mask = pd.Series(False, index=generated_df.index)

    for problem in math500_df["problem"]:
        match_mask |= generated_df["prompt"].str.contains(
            re.escape(problem),
            regex=True,
            na=False
        )

    train_df = generated_df.loc[~match_mask].copy()
    test_df = generated_df.loc[match_mask].copy()
    print(f"moved {len(test_df)} rows to test")
    return train_df, test_df

def remove_think_string(df):
    df["prompt"] = (
        df["prompt"]
        .str.replace(r" Think for \d+ tokens\. <think>", "", regex=True)
        .str.replace(r" Think for \d+ tokens\.", "", regex=True)
    )
    return df


def add_level(generated_df):
    df = pd.read_parquet(
        "hf://datasets/qwedsacf/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
    )
    df = df[["level"]].reset_index().rename(columns={"index": "question_id"})

    # Ensure question_id types match. Cast both to string to be safe.
    df["question_id"] = df["question_id"].astype(str)
    generated_df["question_id"] = generated_df["question_id"].astype(str)

    # Merge level onto generated answers (left join on generated_df)
    merged = generated_df.merge(df, on="question_id", how="left", validate="m:1")

    return merged

def split_math_dataset(generated_df):
    generated_df = add_level(generated_df)
    generated_df = remove_think_string(generated_df)

    dataset = load_dataset("HuggingFaceH4/MATH-500")
    math500_df = dataset['test'].to_pandas()
    train, test= split_math500(generated_df, math500_df)

    return train, test

def split_aimee(aime_df, test_size = 250):
    aime_df["level"] = "aime"
    aime_df = remove_think_string(aime_df)

    unique_question_ids =  aime_df["question_id"].drop_duplicates()
    test_question_ids = unique_question_ids.sample(n=test_size, random_state=KEY).tolist()

    aime_df["split"] = aime_df["question_id"].apply(lambda q: "test" if q in test_question_ids else "train")

    train_df = aime_df[aime_df["split"] == "train"].drop(columns=["split"])
    test_df  = aime_df[aime_df["split"] == "test"].drop(columns=["split"])

    return train_df, test_df

# Creates dataset with columns 
# ['question_id', 'prompt', 'solution_col', 'generated_think_text', 'generated_text', 'target_think_tokens', 
# 'generated_think_tokens', 'latency_sec', 'is_correct', 'level']
def split_datasets(math1_dataset_path, math2_dataset_path, aimee_dataset_path):
    math1 = pd.read_parquet(math1_dataset_path)
    math2 = pd.read_parquet(math2_dataset_path)

    combined_math = pd.concat([math1, math2], ignore_index=True)

    aime = pd.read_parquet(aimee_dataset_path)

    train_df_math, test_df_math = split_math_dataset(combined_math)



    train_df_aimee, test_df_aimee = split_aimee(aime, TEST_SIZE_AIME)

    train_df = pd.concat([train_df_math, train_df_aimee], ignore_index=True)
    test_df = pd.concat([test_df_math, test_df_aimee], ignore_index=True)

    train_df.to_parquet(Path(__file__).parent / "train.parquet", index=False)
    test_df.to_parquet(Path(__file__).parent / "test.parquet", index=False)

split_datasets(MATH1_DATASET_PATH, MATH2_DATASET_PATH, AIME_DATASET_PATH)