import torch


def select_token_indices(
    scores,
    keep_count,
    preserve_edges=True,
    random_selection=False,
    seed=42,
    mandatory_indices=None,
):
    sequence_length = scores.numel()
    keep_count = max(1, min(sequence_length, int(keep_count)))

    # Always preserve the first 4 and last 32 tokens when enabled.
    mandatory = set()
    if preserve_edges and sequence_length > 0:
        mandatory.update(range(min(4, sequence_length)))
        mandatory.update(
            range(max(0, sequence_length - 32), sequence_length)
        )

    # Preserve explicitly requested positions when the budget permits.
    if mandatory_indices is not None:
        mandatory.update(
            int(i)
            for i in mandatory_indices
            if 0 <= int(i) < sequence_length
        )

    # If the mandatory tokens exceed the budget, prioritize the
    # first 4 and then the most recent tokens.
    if len(mandatory) > keep_count:
        first_tokens = list(range(min(4, sequence_length)))
        recent_tokens = list(
            range(max(0, sequence_length - 32), sequence_length)
        )
        selected = list(dict.fromkeys(first_tokens + recent_tokens))
        selected = selected[:keep_count]
        return torch.tensor(
            sorted(selected),
            device=scores.device,
            dtype=torch.long,
        )

    remaining_count = keep_count - len(mandatory)
    if remaining_count == 0:
        return torch.tensor(
            sorted(mandatory),
            device=scores.device,
            dtype=torch.long,
        )

    eligible = torch.ones(
        sequence_length,
        device=scores.device,
        dtype=torch.bool,
    )
    for index in mandatory:
        eligible[index] = False

    eligible_indices = torch.nonzero(
        eligible, as_tuple=False
    ).flatten()

    if random_selection:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutation = torch.randperm(
            eligible_indices.numel(),
            generator=generator,
        )[:remaining_count]
        extra_indices = (
            eligible_indices.cpu()[permutation].to(scores.device)
        )
    else:
        eligible_scores = scores[eligible_indices]
        top_positions = torch.topk(
            eligible_scores,
            k=min(remaining_count, eligible_scores.numel()),
        ).indices
        extra_indices = eligible_indices[top_positions]

    selected = sorted(
        mandatory.union(int(i) for i in extra_indices.tolist())
    )
    return torch.tensor(
        selected,
        device=scores.device,
        dtype=torch.long,
    )