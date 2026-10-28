import os
import pandas as pd
import torch

from src.model_loader import load_model
from src.entropy_calculator import calculate_entropy_and_importance


PROMPT = """
The project note says the important passkey is KEY583104.
The rest of this text contains background details about libraries,
maps, research meetings, engineering notes, and historical documents.
What is the important passkey?
"""


def main():
    tokenizer, model, device = load_model()

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

    entropies, scores, _ = calculate_entropy_and_importance(
        outputs.attentions
    )

    results = pd.DataFrame({
        "layer": range(len(entropies)),
        "normalized_attention_entropy": entropies,
    })

    os.makedirs("results", exist_ok=True)

    results.to_csv(
        "results/entropy_results.csv",
        index=False,
    )

    print("Device:", device)
    print(results.to_string(index=False))
    print("Number of token importance scores:", scores.numel())
    print("Saved to results/entropy_results.csv")


if __name__ == "__main__":
    main()