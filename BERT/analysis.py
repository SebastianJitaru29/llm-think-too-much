import pandas as pd
from pathlib import Path


def split_by_question_type(df):
    aime_mask = df["question_id"].str.contains("aime", case=False, na=False)
    return df[aime_mask], df[~aime_mask]


def compute_metrics(df):
    mean_tokens = df["generated_think_tokens"].mean()
    percent_correct = df["is_correct"].mean() * 100
    return mean_tokens, percent_correct


def main():
    df = pd.read_parquet(Path(__file__).parent.parent / "results.parquet")
    aime_df, math_df = split_by_question_type(df)

    aime_mean_tokens, aime_percent_correct = compute_metrics(aime_df)
    math_mean_tokens, math_percent_correct = compute_metrics(math_df)

    print("AIME mean generated_think_tokens:", aime_mean_tokens)
    print("AIME percent correct:", aime_percent_correct)
    print("Math mean generated_think_tokens:", math_mean_tokens)
    print("Math percent correct:", math_percent_correct)


if __name__ == "__main__":
    main()