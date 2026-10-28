import torch
from src.model_loader import load_model


def main():
    tokenizer, model, device = load_model()

    prompt = "Explain cache memory in one short sentence."

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("Device:", device)

    print(
        tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )
    )

    print("SUCCESS: Model generated text with KV cache enabled.")


if __name__ == "__main__":
    main()