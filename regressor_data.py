from typing import Literal
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from pathlib import Path

from math_equivalence import is_equiv

dynamic_folder = Path(__file__).parent / "dynamic_regressor_results"
static_folder = Path(__file__).parent / "static_regressor_results"

def load_batches(folder: Path) -> pd.DataFrame:

    dfs = []
    for batch_file in folder.glob("*.parquet"):
        
        dfs.append(pd.read_parquet(batch_file))

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)



def dynamic_to_comparison(df: pd.DataFrame) -> pd.DataFrame:
    
    if df.empty:
        return df

    df = df.sort_values(["question_id", "step_i"])
    ignore_cols = ["step_i", "problem"] + df.columns[df.columns.str.contains("bin")].tolist()

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
    ignore_cols = ["problem"] + df.columns[df.columns.str.contains("bin")].tolist()
    out = df.drop(columns=ignore_cols)
    return out

def create_comparison_dataset():

    df = load_batches(static_folder)
    sdf = static_to_comparison(df)
    sdf["method"] = "static"

    df = load_batches(dynamic_folder)
    ddf = dynamic_to_comparison(df)
    ddf["method"] = "dynamic"

    df = pd.concat((sdf, ddf), ignore_index=True)
    
    return df
    # data = Path(__file__).parent / "data"
    # data.mkdir(exist_ok=True)
    
    # df.to_parquet(data / "regressor_comparison.parquet", index=False)

def add_levels_and_solution(df: pd.DataFrame) -> pd.DataFrame:

    data_folder = Path(__file__).parent / "data"

    m = pd.read_parquet(data_folder / "math.parquet", columns=["problem", "solution", "level"])
    m = m.reset_index(names="question_id")
    m["question_id"] = m["question_id"].astype("string")

    a = pd.read_parquet(data_folder / "aime.parquet", columns=["ID", "Question", "Answer"])
    a = a.rename(columns={"ID": "question_id", "Question": "problem", "Answer": "solution"})
    a["level"] = "6"

    q = pd.concat((m, a), ignore_index=True)
    q = q.drop_duplicates("question_id")
    q["level"] = q["level"].str.replace("Level", "").str.strip()
    q["level"] = pd.to_numeric(q["level"], errors="coerce")

    df = pd.merge(df, q[["question_id", "level", "solution"]], "left", "question_id", validate="m:1")
    return df


import re

def extract_boxed(s: str):
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    return m.group(1).strip() if m else None

def evaluate_answer(expected_answer, generated_answer):
    gen_val = extract_boxed(generated_answer)
    if gen_val is None:
        return False

    exp_val = extract_boxed(expected_answer)
    if exp_val is None:
        exp_val = str(expected_answer).strip()

    return is_equiv(gen_val, exp_val)

def fix_aime_correct(df: pd.DataFrame):
    
    print("Fixing aimee is_correct")

    for i in df.index[df["level"] == 6.0]:

        prev = df.loc[i, "is_correct"]
        af = evaluate_answer(df.loc[i, "solution"], df.loc[i, "generated_text"])

        if af != prev:
            print(df.loc[i, "question_id"])
            df.loc[i, "is_correct"] = af


def produce_curves(
    alldf: pd.DataFrame,
    scatter_points: Literal["x", "numbers"],
    fit_line_degree: int  = 2
):
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim([-0.1, 1.1])

    for i, (method, df) in enumerate(alldf.groupby("method")):
        tc = df.groupby("level")["actual_tokens"].mean().sort_index()
        acc = df.groupby("level")["is_correct"].mean().sort_index()
        assert (tc.index == acc.index).all()
        color = colors[i]

        for level, x, y in zip(tc.index, tc, acc):

            if scatter_points == "x":
                ax.plot(x, y, marker='x', color=color, markersize=8, markeredgewidth=2)

            elif scatter_points == "numbers":
                ax.text(
                    x, y, str(int(level)),
                    color=color, fontsize=12, fontweight='bold',
                    ha='center', va='center',
                    path_effects=[pe.withStroke(linewidth=5, foreground="black")]
                )

        coeffs = np.polyfit(tc.astype(float).to_numpy(), acc.astype(float).to_numpy(), deg=fit_line_degree)
        poly = np.poly1d(coeffs)

        x_fit = np.linspace(tc.min(), tc.max(), 200)
        y_fit = poly(x_fit)
        ax.plot(x_fit, y_fit, color=color, label=method, linewidth=3)


    ax.set_xlabel("Actual token counts", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.legend()

    fig.tight_layout()
    fig.savefig("./curves.pdf")

    plt.show()
        

def load_non_regressor_data() -> pd.DataFrame:
    df = pd.read_csv(Path(__file__).parent / "data" / "results.csv", header=None)
    df.columns = ["question_id", "level", "actual_tokens", "is_correct", "method"]

    df["level"] = df["level"].str.replace("Level", "").str.replace("aimee", "6").str.strip()
    df["level"] = pd.to_numeric(df["level"], errors="raise")

    df = df.sort_values(["method", "level"])
    
    return df


if __name__ == "__main__":
    df = create_comparison_dataset()
    df = add_levels_and_solution(df)
    fix_aime_correct(df)

    other = load_non_regressor_data()
    df = pd.concat((df, other), ignore_index=True)
    produce_curves(
        df,
        scatter_points="numbers",
        fit_line_degree=2
    )
