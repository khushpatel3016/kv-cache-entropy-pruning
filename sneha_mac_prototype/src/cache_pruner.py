import torch


def cache_sequence_length(cache):
    try:
        return int(cache.get_seq_length())
    except Exception:
        pass

    if hasattr(cache, "key_cache") and cache.key_cache:
        return int(cache.key_cache[0].shape[-2])

    if hasattr(cache, "layers") and cache.layers:
        return int(cache.layers[0].keys.shape[-2])

    return 0


def prune_cache(cache, keep_indices):
    if cache is None:
        raise ValueError("No KV cache was returned by the model.")

    keep_indices = keep_indices.to(dtype=torch.long)

    if hasattr(cache, "layers"):
        layer_count = len(cache.layers)

        for layer_index in range(layer_count):
            layer = cache.layers[layer_index]

            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)

            if keys is None or values is None:
                raise TypeError(
                    f"Cache tensors unavailable in layer {layer_index}."
                )

            key_indices = keep_indices.to(keys.device)
            value_indices = keep_indices.to(values.device)

            layer.keys = keys.index_select(
                -2, key_indices
            ).contiguous()

            layer.values = values.index_select(
                -2, value_indices
            ).contiguous()

    elif hasattr(cache, "key_cache"):
        layer_count = len(cache.key_cache)

        for layer_index in range(layer_count):
            keys = cache.key_cache[layer_index]
            values = cache.value_cache[layer_index]

            cache.key_cache[layer_index] = keys.index_select(
                -2, keep_indices.to(keys.device)
            ).contiguous()

            cache.value_cache[layer_index] = values.index_select(
                -2, keep_indices.to(values.device)
            ).contiguous()

    else:
        raise TypeError(
            "Unsupported KV cache format. "
            "Check your Transformers version."
        )

    return cache