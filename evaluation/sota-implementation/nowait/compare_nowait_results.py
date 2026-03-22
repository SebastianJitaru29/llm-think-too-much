"""
Compare NOWAIT replication results across all runs.

Usage:
    python compare_nowait_results.py --results-dir ./eval_results
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="./eval_results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    
    # Find all summary files
    summaries = []
    for summary_file in sorted(results_dir.glob("*/evaluation_summary.parquet")):
        run_name = summary_file.parent.name
        df = pd.read_parquet(summary_file)
        df["run"] = run_name
        summaries.append(df)

    if not summaries:
        print(f"No results found in {results_dir}")
        return

    all_results = pd.concat(summaries, ignore_index=True)

    # Pivot: rows = dataset, columns = run, values = accuracy / avg_tokens
    datasets = all_results["dataset"].unique()
    runs = all_results["run"].unique()

    # Print accuracy table
    print(f"\n{'='*90}")
    print(f"  ACCURACY COMPARISON (%)")
    print(f"{'='*90}")
    header = f"{'Dataset':<15}"
    for run in runs:
        # Shorten run name for display
        short = run.replace("DeepSeek-R1-Distill-Qwen-1.5B_", "")
        header += f" {short:>18}"
    print(header)
    print(f"{'-'*90}")

    for dataset in sorted(datasets):
        row = f"{dataset:<15}"
        for run in runs:
            match = all_results[(all_results["dataset"] == dataset) & (all_results["run"] == run)]
            if len(match) > 0:
                acc = match.iloc[0]["accuracy"] * 100
                row += f" {acc:>17.2f}%"
            else:
                row += f" {'—':>18}"
        print(row)

    # Print token table
    print(f"\n{'='*90}")
    print(f"  AVG TOKENS COMPARISON")
    print(f"{'='*90}")
    header = f"{'Dataset':<15}"
    for run in runs:
        short = run.replace("DeepSeek-R1-Distill-Qwen-1.5B_", "")
        header += f" {short:>18}"
    print(header)
    print(f"{'-'*90}")

    for dataset in sorted(datasets):
        row = f"{dataset:<15}"
        for run in runs:
            match = all_results[(all_results["dataset"] == dataset) & (all_results["run"] == run)]
            if len(match) > 0:
                tokens = match.iloc[0]["avg_tokens"]
                row += f" {tokens:>18.1f}"
            else:
                row += f" {'—':>18}"
        print(row)

    # Print token reduction relative to baseline
    baseline_run = [r for r in runs if "baseline" in r]
    if baseline_run:
        baseline_run = baseline_run[0]
        nowait_runs = [r for r in runs if r != baseline_run]

        print(f"\n{'='*90}")
        print(f"  TOKEN REDUCTION vs BASELINE (%)")
        print(f"{'='*90}")
        header = f"{'Dataset':<15}"
        for run in nowait_runs:
            short = run.replace("DeepSeek-R1-Distill-Qwen-1.5B_", "")
            header += f" {short:>18}"
        print(header)
        print(f"{'-'*90}")

        for dataset in sorted(datasets):
            row = f"{dataset:<15}"
            base = all_results[
                (all_results["dataset"] == dataset) & (all_results["run"] == baseline_run)
            ]
            if len(base) == 0:
                continue
            base_tokens = base.iloc[0]["avg_tokens"]

            for run in nowait_runs:
                match = all_results[
                    (all_results["dataset"] == dataset) & (all_results["run"] == run)
                ]
                if len(match) > 0:
                    tokens = match.iloc[0]["avg_tokens"]
                    reduction = (1 - tokens / base_tokens) * 100 if base_tokens > 0 else 0
                    row += f" {reduction:>17.1f}%"
                else:
                    row += f" {'—':>18}"
            print(row)

    # Print accuracy delta vs baseline
    if baseline_run:
        print(f"\n{'='*90}")
        print(f"  ACCURACY DELTA vs BASELINE (pp)")
        print(f"{'='*90}")
        header = f"{'Dataset':<15}"
        for run in nowait_runs:
            short = run.replace("DeepSeek-R1-Distill-Qwen-1.5B_", "")
            header += f" {short:>18}"
        print(header)
        print(f"{'-'*90}")

        for dataset in sorted(datasets):
            row = f"{dataset:<15}"
            base = all_results[
                (all_results["dataset"] == dataset) & (all_results["run"] == baseline_run)
            ]
            if len(base) == 0:
                continue
            base_acc = base.iloc[0]["accuracy"] * 100

            for run in nowait_runs:
                match = all_results[
                    (all_results["dataset"] == dataset) & (all_results["run"] == run)
                ]
                if len(match) > 0:
                    acc = match.iloc[0]["accuracy"] * 100
                    delta = acc - base_acc
                    sign = "+" if delta >= 0 else ""
                    row += f" {sign}{delta:>16.2f}pp"
                else:
                    row += f" {'—':>18}"
            print(row)

    print(f"\n{'='*90}")
    
    # Save combined results
    output_file = results_dir / "nowait_comparison.csv"
    all_results.to_csv(output_file, index=False)
    print(f"Saved combined results to {output_file}")


if __name__ == "__main__":
    main()