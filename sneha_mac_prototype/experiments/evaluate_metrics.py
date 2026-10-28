
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Summarize KV-cache pruning experiment results."
    )

    parser.add_argument("--input", required=True, help="Input results CSV")
    parser.add_argument(
        "--method-col", default="method",
        help="Column containing the method name"
    )
    parser.add_argument(
        "--accuracy-col", default="accuracy",
        help="Column containing accuracy values"
    )
    parser.add_argument(
        "--latency-col", default="latency_seconds",
        help="Column containing latency in seconds"
    )
    parser.add_argument(
        "--cache-col", default="cache_entries",
        help="Column containing cache-entry counts"
    )
    parser.add_argument(
        "--output", default="results/metrics_summary.csv",
        help="Output summary CSV"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("The input CSV contains no rows.")

    columns = [
        args.method_col,
        args.accuracy_col,
        args.latency_col,
        args.cache_col,
    ]

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Pass the correct column names using the command-line options."
        )

    for column in [args.accuracy_col, args.latency_col, args.cache_col]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    summary = (
        df.groupby(args.method_col, dropna=False)
        .agg(
            trials=(args.accuracy_col, "count"),
            mean_accuracy=(args.accuracy_col, "mean"),
            mean_latency_seconds=(args.latency_col, "mean"),
            mean_cache_entries=(args.cache_col, "mean"),
        )
        .reset_index()
    )

    # Calculate cache reduction relative to the full-cache baseline,
    # if a method is explicitly named "full" or "full_cache".
    method_names = summary[args.method_col].astype(str).str.lower()
    baseline = summary[
        method_names.isin(["full", "full_cache", "full cache"])
    ]

    if not baseline.empty:
        baseline_cache = baseline.iloc[0]["mean_cache_entries"]

        if pd.notna(baseline_cache) and baseline_cache > 0:
            summary["cache_reduction_percent"] = (
                1 - summary["mean_cache_entries"] / baseline_cache
            ) * 100

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print("\nExperiment summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
