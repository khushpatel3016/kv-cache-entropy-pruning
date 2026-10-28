import os
import sys
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, REPO)

from src.model_loader import load_model
from src.cache_pruner import prune_cache, cache_sequence_length
from src.entropy_calculator import calculate_entropy_and_importance
from src.token_selector import select_token_indices
from experiments.passkey_benchmark import make_prompt

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 32
PASSKEY = "KEY123456"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    print("Device:", device)
    print("Dtype:", torch.float32 if device.type == "cpu" else torch.float16)

    tokenizer, model, device = load_model(MODEL_NAME, device)
    prompt = make_prompt(PASSKEY, filler_sentences=18)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)
    input_ids = inputs["input_ids"]
    input_length = input_ids.shape[1]
    print("Prompt tokens:", input_length)
    print("Passkey present in prompt:", PASSKEY in prompt)

    # A: ordinary full-cache generation
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_ids = output[0, input_length:]
    full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    print("\n=== FULL / model.generate ===")
    print("Text:", repr(full_text))
    print("First 10 generated IDs:", full_ids[:10].tolist())

    # B/C: same manual decoding loop, varying only retained cache budget
    for budget in (1.0, 0.5):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=inputs.get("attention_mask"),
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        entropies, importance_scores, _ = calculate_entropy_and_importance(
            outputs.attentions
        )
        cache = outputs.past_key_values
        before = cache_sequence_length(cache)
        keep_count = min(input_length, max(2, round(input_length * budget)))
        keep_indices = select_token_indices(
            importance_scores,
            keep_count,
            preserve_edges=True,
            random_selection=False,
            seed=42,
        )
        print(f"\n=== FIXED / budget={budget} ===")
        print("Cache length before prune:", before)
        print("Keep count:", keep_count)
        print("First 20 kept positions:", keep_indices[:20].tolist())

        encoded_offsets = tokenizer(
            prompt,
            return_offsets_mapping=True,
            truncation=True,
            max_length=1024,
        )
        char_start = prompt.index(PASSKEY)
        char_end = char_start + len(PASSKEY)
        passkey_positions = [
            i for i, (a, b) in enumerate(encoded_offsets["offset_mapping"])
            if a < char_end and b > char_start
        ]

        kept_set = set(keep_indices.tolist())
        print("Passkey character span:", (char_start, char_end))
        print("Passkey token positions:", passkey_positions)
        print("Passkey positions retained:",
              [pos for pos in passkey_positions if pos in kept_set])
        print("Passkey tokens all retained:",
              bool(passkey_positions)
              and all(pos in kept_set for pos in passkey_positions))

        print("Last 32 positions all kept:",
              set(range(max(0, input_length - 32), input_length))
              .issubset(set(keep_indices.tolist())))

        cache = prune_cache(cache, keep_indices)
        after = cache_sequence_length(cache)
        print("Cache length after prune:", after)

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token]
        current_position = input_length
        past = cache

        for _ in range(MAX_NEW_TOKENS - 1):
            past_length = cache_sequence_length(past)
            attention_mask = torch.ones(
                (1, past_length + 1), dtype=torch.long, device=device
            )
            position_ids = torch.tensor(
                [[current_position]], dtype=torch.long, device=device
            )
            cache_position = torch.tensor(
                [past_length], dtype=torch.long, device=device
            )
            with torch.no_grad():
                step_output = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = step_output.past_key_values
            next_token = step_output.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            generated.append(next_token)
            current_position += 1
            stop_ids = {
                token_id
                for token_id in (
                    tokenizer.eos_token_id,
                    tokenizer.pad_token_id,
                )
                if token_id is not None
            }
            if next_token.item() in stop_ids:
                break

        generated_ids = torch.cat(generated, dim=1)[0]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("Text:", repr(text))
        print("First 10 generated IDs:", generated_ids[:10].tolist())

if __name__ == "__main__":
    main()
