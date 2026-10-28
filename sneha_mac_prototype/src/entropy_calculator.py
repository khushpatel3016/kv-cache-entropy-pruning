import math
import torch


def calculate_entropy_and_importance(attentions):
    if not attentions:
        raise ValueError(
            "No attention maps returned. "
            "Make sure eager attention is enabled."
        )

    layer_entropies = []
    layer_importances = []

    for attention in attentions:
        # Attention shape: batch, heads, query_tokens, key_tokens
        attention = attention.float().clamp_min(1e-12)

        # Calculate entropy over the last 32 query positions.
        recent_attention = attention[..., -32:, :]

        entropy = -(
            recent_attention * recent_attention.log()
        ).sum(dim=-1).mean()

        sequence_length = attention.shape[-1]
        normalized_entropy = (
            entropy.item() / math.log(max(sequence_length, 2))
        )
        normalized_entropy = max(
            0.0, min(1.0, normalized_entropy)
        )
        layer_entropies.append(normalized_entropy)

        # Estimate token importance from recent query positions.
        importance = recent_attention.mean(dim=(0, 1, 2))
        importance = importance / importance.sum().clamp_min(1e-12)
        layer_importances.append(importance)

    stacked_importances = torch.stack(layer_importances, dim=0)

    entropy_weights = torch.tensor(
        layer_entropies,
        device=stacked_importances.device,
        dtype=stacked_importances.dtype,
    )
    entropy_weights = entropy_weights + 1e-3
    entropy_weights = entropy_weights / entropy_weights.sum()

    combined_importance = (
        stacked_importances * entropy_weights[:, None]
    ).sum(dim=0)

    combined_importance = (
        combined_importance
        / combined_importance.sum().clamp_min(1e-12)
    )

    return (
        layer_entropies,
        combined_importance,
        layer_importances,
    )