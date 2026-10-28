
from datasets import load_dataset

TASKS = ["qasper", "hotpotqa"]

for task in TASKS:
    print(f"\nLoading LongBench task: {task}")

    dataset = load_dataset(
        "THUDM/LongBench",
        task,
        split="test"
    )

    print("Number of examples:", len(dataset))
    print("Columns:", dataset.column_names)

    example = dataset[0]

    print("Context preview:", example["context"][:300])
    print("Question:", example["input"])
    print("Expected answer:", example["answers"][:2])
