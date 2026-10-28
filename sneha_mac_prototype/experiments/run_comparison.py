import os
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt


def main():
    os.makedirs("results", exist_ok=True)

    command = [
        sys.executable,
        "experiments/passkey_benchmark.py",
    ]

    print("Running benchmark...")

    subprocess.run(
        command,
        check=True,
    )

    summary_path = "results/summary.csv"

    if not os.path.exists(summary_path):
        print("No summary file was created.")
        return

    results = pd.read_csv(summary_path)

    if results.empty:
        print("The summary file is empty.")
        return

    # Graph 1: Retrieval accuracy
    plt.figure()

    plt.bar(
        results["policy"],
        results["retrieval_accuracy"],
    )

    plt.xlabel("Cache policy")
    plt.ylabel("Retrieval accuracy")
    plt.ylim(0, 1)
    plt.title("Passkey Retrieval Accuracy")

    plt.tight_layout()

    plt.savefig(
        "results/retrieval_accuracy.png",
        dpi=160,
    )

    plt.close()

    # Graph 2: Cache reduction
    plt.figure()

    plt.bar(
        results["policy"],
        results["mean_cache_reduction_pct"],
    )

    plt.xlabel("Cache policy")
    plt.ylabel("Cache reduction (%)")
    plt.title("KV-Cache Token Reduction")

    plt.tight_layout()

    plt.savefig(
        "results/cache_reduction.png",
        dpi=160,
    )

    plt.close()

    # Graph 3: Execution time
    plt.figure()

    plt.bar(
        results["policy"],
        results["mean_elapsed_seconds"],
    )

    plt.xlabel("Cache policy")
    plt.ylabel("Elapsed time (seconds)")
    plt.title("Inference Time Comparison")

    plt.tight_layout()

    plt.savefig(
        "results/latency.png",
        dpi=160,
    )

    plt.close()

    print("\nAll graphs generated successfully.")
    print("Check the results folder for:")
    print("1. retrieval_accuracy.png")
    print("2. cache_reduction.png")
    print("3. latency.png")

    print(
        "\nReport speed or accuracy improvements only "
        "if your experimental results support them."
    )


if __name__ == "__main__":
    main()