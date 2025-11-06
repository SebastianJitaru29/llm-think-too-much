import numpy as np
import pandas as pd
from pathlib import Path

dynamic_folder = Path(__file__).parent / "dynamic_regressor_results"
static_folder = Path(__file__).parent / "static_regressor_results"

def load_batches(folder: Path) -> pd.DataFrame:

    dfs = []
    for batch_file in folder.glob("*.parquet"):
        
        dfs.append(pd.read_parquet(batch_file))

    return pd.concat(dfs, ignore_index=True)


def dynamic_to_comparison(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.sort_values(["question_id", "step_i"])
    ignore_cols = ["step_i", "problem", "expected_solution"] + df.columns[df.columns.str.contains("bin")].tolist()

    out = []

    for qid, qdf in df.groupby("question_id"):

        correct = qdf["is_correct"].unique()
        correct = [c for c in correct if (c is not None)]

        assert len(correct) < 2, correct

        # Did not answer within limits
        if len(correct) == 0:
            row = qdf.iloc[-1, :]
        else:
            (inds,) = np.where(qdf["is_correct"] == correct[0])
            i = inds.min()
            row = qdf.iloc[i, :]

        row = row.drop(ignore_cols)
        out.append(row.to_dict())


    out = pd.DataFrame(out)
    return out

def static_to_comparison(df: pd.DataFrame) -> pd.DataFrame:
    ignore_cols = ["problem", "expected_solution"] + df.columns[df.columns.str.contains("bin")].tolist()
    out = df.drop(columns=ignore_cols)
    return out

def create_comparison_dataset():

    df = load_batches(static_folder)
    sdf = static_to_comparison(df)
    sdf["type"] = "static"

    df = load_batches(dynamic_folder)
    ddf = dynamic_to_comparison(df)
    ddf["type"] = "dynamic"

    df = pd.concat((sdf, ddf), ignore_index=True)
    data = Path(__file__).parent / "data"
    data.mkdir(exist_ok=True)
    
    df.to_parquet(data / "regressor_comparison.parquet", index=False)

if __name__ == "__main__":
    create_comparison_dataset()