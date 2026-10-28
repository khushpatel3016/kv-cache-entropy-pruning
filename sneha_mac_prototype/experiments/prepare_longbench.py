
from datasets import load_dataset
import json
import os

TASKS = ["qasper", "hotpotqa"]
SAMPLES_PER_TASK = 10

os.makedirs("data/longbench", exist_ok=True)

for task in TASKS:
    print(f"\nLoading {task}...")

    dataset = load_dataset(
        "THUDM/LongBench",
        task,
        split="test"
    )

    count = min(SAMPLES_PER_TASK, len(dataset))
    samples = []

    for i in range(count):
        item = dataset[i]

        samples.append({
            "task": task,
            "id": item.get("_id", str(i)),
            "context": item["context"],
            "question": item["input"],
            "answers": item["answers"],
            "length": item.get("length", None)
        })

    output_path = f"data/longbench/{task}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Saved {count} examples to {output_path}")

print("\nLongBench sample preparation complete.")
