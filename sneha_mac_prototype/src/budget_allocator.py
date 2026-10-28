def entropy_guided_ratio(
    layer_entropies,
    minimum=0.25,
    maximum=0.75,
):
    if not layer_entropies:
        return maximum

    mean_entropy = (
        sum(layer_entropies)
        / len(layer_entropies)
    )

    mean_entropy = max(
        0.0,
        min(1.0, mean_entropy)
    )

    ratio = minimum + (
        maximum - minimum
    ) * mean_entropy

    return ratio


def fixed_ratio(ratio=0.5):
    return max(
        0.05,
        min(1.0, float(ratio))
    )