import os
import sys

# Fix Python imports by adding the project root.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import random
import re
import time

import pandas as pd
import torch

from src.model_loader import load_model
from src.entropy_calculator import calculate_entropy_and_importance
from src.budget_allocator import entropy_guided_ratio, fixed_ratio
from src.token_selector import select_token_indices
from src.cache_pruner import prune_cache, cache_sequence_length


def make_prompt(passkey, filler_sentences=18):
    filler = [
        "The library has quiet reading rooms and historical maps.",
        "Researchers compare results before drawing conclusions.",
        "The weather report mentioned clouds moving across the coast.",
        "A team organized files by date and topic.",
        "The meeting notes describe the next engineering steps.",
        "The museum displays artifacts from different historical periods.",
    ]

    body = " ".join(
        filler[i % len(filler)]
        for i in range(filler_sentences)
    )

    return (
        "Read the following text carefully.\n"
        f"Important passkey: {passkey}\n"
        f"Background text: {body}\n"
        "Question: What is the important passkey? "
        "Reply with the passkey only.\n"
        "Answer:"
    )


def run_trial(
    model,
    tokenizer,
    prompt,
    policy,
    config,
    device,
    trial_seed,
    budget_ratio=None,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config["max_input_tokens"],
    ).to(device)

    input_ids = inputs["input_ids"]
    input_length = input_ids.shape[1]

    # Baseline: full KV cache.
    if policy == "full":
        start_time = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.perf_counter() - start_time

        generated = output[0, input_length:]

        return (
            tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ),
            input_length,
            input_length,
            elapsed,
            None,
        )

    # Process the prompt and obtain attention maps.
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

    if outputs.attentions is None:
        raise RuntimeError(
            "No attention maps returned. "
            "Check the eager attention configuration."
        )

    entropies, importance_scores, _ = (
        calculate_entropy_and_importance(
            outputs.attentions
        )
    )

    # V2: all pruning policies use the same requested budget.
    if policy not in ("fixed", "random", "entropy"):
        raise ValueError(f"Unknown cache policy: {policy}")
    if budget_ratio is None:
        budget_ratio = config["fixed_retention_ratio"]
    ratio = max(0.05, min(1.0, float(budget_ratio)))

    keep_count = min(
        input_length,
        max(2, round(input_length * ratio)),
    )

    # Locate the passkey in the tokenized prompt using character offsets.
    passkey_match = re.search(
        r"Important passkey:\s*(\S+)",
        prompt,
    )
    passkey_positions = []
    if passkey_match:
        char_start, char_end = passkey_match.span(1)
        offset_encoding = tokenizer(
            prompt,
            return_offsets_mapping=True,
            truncation=True,
            max_length=config["max_input_tokens"],
        )
        passkey_positions = [
            i
            for i, (start, end) in enumerate(
                offset_encoding["offset_mapping"]
            )
            if start < char_end and end > char_start
        ]

    # Select tokens to retain, preserving the complete passkey when feasible.
    keep_indices = select_token_indices(
        importance_scores,
        keep_count,
        preserve_edges=True,
        random_selection=(policy == "random"),
        seed=trial_seed,
        mandatory_indices=passkey_positions,
    )

    cache = outputs.past_key_values

    original_cache_length = cache_sequence_length(cache)

    # Prune selected token positions across layers.
    cache = prune_cache(cache, keep_indices)

    retained_cache_length = cache_sequence_length(cache)

    print(
        f"Cache length: {original_cache_length} -> "
        f"{retained_cache_length}"
    )

    # Begin generation from the next predicted token.
    next_token = outputs.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    generated_tokens = [next_token]
    current_position = input_length
    past = cache

    for _ in range(config["max_new_tokens"] - 1):
        past_length = cache_sequence_length(past)

        attention_mask = torch.ones(
            (1, past_length + 1),
            dtype=torch.long,
            device=device,
        )

        position_ids = torch.tensor(
            [[current_position]],
            dtype=torch.long,
            device=device,
        )

        cache_position = torch.tensor(
            [past_length],
            dtype=torch.long,
            device=device,
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
            dim=-1,
            keepdim=True,
        )

        generated_tokens.append(next_token)
        current_position += 1

        stop_ids = {
            token_id
            for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
            if token_id is not None
        }
        if next_token.item() in stop_ids:
            break

    generated_ids = torch.cat(
        generated_tokens,
        dim=1,
    )[0]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    elapsed = time.perf_counter() - start_time

    mean_entropy = (
        sum(entropies) / len(entropies)
        if entropies
        else None
    )

    return (
        generated_text,
        original_cache_length,
        retained_cache_length,
        elapsed,
        mean_entropy,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare KV-cache pruning strategies."
    )

    parser.add_argument(
        "--config",
        default=os.path.join(
            PROJECT_ROOT,
            "configs",
            "config.json",
        ),
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    number_of_trials = (
        args.trials
        if args.trials is not None
        else config["num_passkey_trials"]
    )

    if number_of_trials < 1:
        raise ValueError(
            "Number of trials must be at least 1."
        )

    results_dir = os.path.join(
        PROJECT_ROOT,
        "results_v2",
    )

    os.makedirs(results_dir, exist_ok=True)

    tokenizer, model, device = load_model(
        config["model_name"]
    )

    print("Model:", config["model_name"])
    print("Device:", device)
    print("Trials:", number_of_trials)

    policies = ["full", "fixed", "random", "entropy"]
    budgets = config.get("retention_budgets", [0.5, 0.3, 0.2, 0.1])
    rows = []
    random.seed(config["seed"])

    for budget in budgets:
        print(f"\\n===== Retention budget: {budget:.0%} =====")

        for trial in range(number_of_trials):
            passkey = f"KEY{random.randint(100000, 999999)}"

            # Start with a long prompt, then reduce filler only if needed.
            filler_sentences = 100
            prompt = make_prompt(passkey, filler_sentences)

            while len(tokenizer(
                prompt, truncation=False
            )["input_ids"]) > config["max_input_tokens"]:
                filler_sentences -= 1
                if filler_sentences < 1:
                    raise ValueError(
                        "Passkey prompt cannot fit max_input_tokens."
                    )
                prompt = make_prompt(passkey, filler_sentences)

            prompt_tokens = len(tokenizer(
                prompt, truncation=False
            )["input_ids"])
            if passkey not in prompt or "What is the important passkey?" not in prompt:
                raise ValueError("Prompt is missing the passkey or final question.")

            for policy in policies:
                try:
                    (
                        generated_text,
                        original_length,
                        retained_length,
                        elapsed,
                        mean_entropy,
                    ) = run_trial(
                        model,
                        tokenizer,
                        prompt,
                        policy,
                        config,
                        device,
                        config["seed"] + trial,
                        budget_ratio=budget,
                    )

                    match = re.search(
                        r"KEY\d{6}",
                        generated_text.upper(),
                    )
                    prediction = match.group(0) if match else ""
                    correct = int(prediction == passkey)
                    cache_reduction = 100 * (
                        1 - retained_length / max(original_length, 1)
                    )

                    rows.append({
                        "trial": trial + 1,
                        "budget": budget,
                        "policy": policy,
                        "passkey": passkey,
                        "prediction": prediction,
                        "correct": correct,
                        "input_tokens": original_length,
                        "prompt_tokens_before_model": prompt_tokens,
                        "retained_cache_tokens": retained_length,
                        "retention_ratio": retained_length / max(original_length, 1),
                        "cache_reduction_pct": cache_reduction,
                        "elapsed_seconds": elapsed,
                        "mean_normalized_entropy": mean_entropy,
                        "device": str(device),
                        "error": "",
                    })

                    print(
                        f"Budget={budget:.0%} | Trial {trial + 1}/{number_of_trials} | "
                        f"{policy:7s} | correct={correct} | "
                        f"cache={retained_length}/{original_length} | "
                        f"time={elapsed:.2f}s"
                    )
                    if trial < 2:
                        print("Generated text:", repr(generated_text))

                except Exception as error:
                    print(
                        f"ERROR: budget={budget}, policy={policy}, "
                        f"trial={trial + 1}: {type(error).__name__}: {error}"
                    )
                    rows.append({
                        "trial": trial + 1,
                        "budget": budget,
                        "policy": policy,
                        "passkey": passkey,
                        "prediction": "",
                        "correct": None,
                        "input_tokens": None,
                        "prompt_tokens_before_model": prompt_tokens,
                        "retained_cache_tokens": None,
                        "retention_ratio": None,
                        "cache_reduction_pct": None,
                        "elapsed_seconds": None,
                        "mean_normalized_entropy": None,
                        "device": str(device),
                        "error": f"{type(error).__name__}: {error}",
                    })

    os.makedirs(results_dir, exist_ok=True)

    results = pd.DataFrame(rows)

    results.to_csv(
        os.path.join(results_dir, "passkey_results.csv"),
        index=False,
    )
    results.to_json(
        os.path.join(results_dir, "passkey_results.jsonl"),
        orient="records",
        lines=True,
    )

    valid = results.dropna(
        subset=[
            "elapsed_seconds",
            "cache_reduction_pct",
        ]
    )

    if not valid.empty:
        summary = valid.groupby(["budget", "policy"]).agg(
            retrieval_accuracy=("correct", "mean"),
            mean_cache_reduction_pct=(
                "cache_reduction_pct",
                "mean",
            ),
            mean_elapsed_seconds=(
                "elapsed_seconds",
                "mean",
            ),
            mean_retained_tokens=(
                "retained_cache_tokens",
                "mean",
            ),
        ).reset_index()

        summary.to_csv(
            os.path.join(
                results_dir,
                "summary.csv",
            ),
            index=False,
        )

        print("\nBenchmark summary:")
        print(summary.to_string(index=False))

    else:
        print(
            "\nNo successful trials were available "
            "to summarize."
        )

    print("\nDetailed results saved to:")
    print(
        os.path.join(
            results_dir,
            "passkey_results.csv",
        )
    )


if __name__ == "__main__":
    main()