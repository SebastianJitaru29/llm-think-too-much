import pandas as pd 
import os
import csv
from pathlib import Path
import re

MATH_DATASET_PATHS = "/home3/s3799042/data_nlp/MATH"
AIME_DATASET_PATH = "/home3/s3799042/data_nlp/AIMEE"

def get_generated_dataset(data_folder):
    generated_dataset = []
    for root, dirs, files in os.walk(data_folder):
        for dir in dirs:
            if "generated" not in dir:
                continue
            path_folder = data_folder + "/" + dir
            for file in os.listdir(path_folder):
                filepath = os.path.join(path_folder, file)
                with open(filepath, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        generated_dataset.append(row)
    
    return generated_dataset            


def remove_think_string(df):
    for _, row in df.iterrows():
        row['prompt'] = re.sub(r" Think for \d+ tokens\. \<think\>", "", row['prompt'])
        row['prompt'] = re.sub(r" Think for \d+ tokens\.", "", row['prompt'])

def split_math_dataset(math_dataset_paths, test_size_per_level=20, random_state=42):
    # load original df (index is the unique question id)
    df = pd.read_parquet(
        "hf://datasets/qwedsacf/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
    )
    # keep only needed column and turn index into a column named question_id
    df = df[["level"]].reset_index().rename(columns={"index": "question_id"})

    # read and concat generated datasets
    #gen_dfs = []
    gen = get_generated_dataset(math_dataset_paths)
    generated_df = pd.DataFrame(gen)

    #generated_df = pd.concat(gen_dfs, ignore_index=True)
    remove_think_string(generated_df)

    # Ensure question_id types match. Cast both to string to be safe.
    df["question_id"] = df["question_id"].astype(str)
    generated_df["question_id"] = generated_df["question_id"].astype(str)

    # Merge level onto generated answers (left join on generated_df)
    merged = generated_df.merge(df, on="question_id", how="left", validate="m:1")

    # Sanity check: any missing levels?
    missing_levels = merged["level"].isna().sum()
    if missing_levels:
        raise ValueError(f"{missing_levels} generated rows have no matching question in df")

    # choose unique questions per level for the test set
    question_levels = merged[["question_id", "level"]].drop_duplicates()

    def sample_per_level(grp):
        n = min(test_size_per_level, len(grp))
        return grp.sample(n=n, random_state=random_state)

    test_questions = question_levels.groupby("level", group_keys=False).apply(sample_per_level)
    test_qids = set(test_questions["question_id"].tolist())

    # assign split so all answers for a question go to same split
    merged["split"] = merged["question_id"].apply(lambda q: "test" if q in test_qids else "train")

    train_df = merged[merged["split"] == "train"].drop(columns=["split"])
    test_df  = merged[merged["split"] == "test"].drop(columns=["split"])

    return train_df, test_df

def split_aimee(path_aimee, test_size = 20, random_state = 42):
    gen = get_generated_dataset(path_aimee)
    gen_df = pd.DataFrame(gen)
    gen_df["level"] = "aimee"
    remove_think_string(gen_df)

    unique_question_ids =  gen_df["question_id"].drop_duplicates()
    test_question_ids = unique_question_ids.sample(n=test_size, random_state=random_state).tolist()

    gen_df["split"] = gen_df["question_id"].apply(lambda q: "test" if q in test_question_ids else "train")

    train_df = gen_df[gen_df["split"] == "train"].drop(columns=["split"])
    test_df  = gen_df[gen_df["split"] == "test"].drop(columns=["split"])

    return train_df, test_df

# Creates dataset with columns 
# ['question_id', 'prompt', 'solution_col', 'generated_think_text', 'generated_text', 'target_think_tokens', 
# 'generated_think_tokens', 'latency_sec', 'is_correct', 'level']
def split_datasets(math_dataset_paths, aimee_dataset_path):
    train_df_math, test_df_math = split_math_dataset(math_dataset_paths)
    train_df_aimee, test_df_aimee = split_aimee(aimee_dataset_path)

    train_df = pd.concat([train_df_math, train_df_aimee], ignore_index=True)
    test_df = pd.concat([test_df_math, test_df_aimee], ignore_index=True)

    org_len_train = len(train_df)
    org_len_test =  len(test_df)

    train_df = train_df.drop_duplicates(subset=['question_id', 'target_think_tokens'], keep='first')
    test_df = test_df.drop_duplicates(subset=['question_id', 'target_think_tokens'], keep='first')

    print(f"Filtered train: {org_len_train - len(train_df)} test: {org_len_test - len(test_df)}")

    train_df.to_parquet(Path(__file__).parent / "train.parquet", index=False)
    test_df.to_parquet(Path(__file__).parent / "test.parquet", index=False)

split_datasets(MATH_DATASET_PATHS, AIME_DATASET_PATH)