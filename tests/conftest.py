"""Shared fixtures: tiny randomly initialised decoder models and a small tokenizer.

These run on CPU in seconds with no downloads. Query/key projections are
scaled up so attention is peaked (as in trained models) rather than uniform,
which gives layers genuinely different entropies.
"""

from __future__ import annotations

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvprune.attention import register_kvprune_attention  # noqa: E402
from kvprune.data import passkey as pk  # noqa: E402


def make_tiny(arch: str = "qwen2", layers: int = 4, sharpen: float = 6.0, seed: int = 0, attn: str = "kvprune"):
    from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config

    register_kvprune_attention()
    torch.manual_seed(seed)
    Cfg = {"llama": LlamaConfig, "qwen2": Qwen2Config}[arch]
    cfg = Cfg(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
              bos_token_id=1, eos_token_id=2, pad_token_id=0)
    m = AutoModelForCausalLM.from_config(copy.deepcopy(cfg), attn_implementation=attn).eval()
    with torch.no_grad():
        for i, layer in enumerate(m.model.layers):
            s = sharpen * (0.5 + i / max(1, layers - 1))  # different sharpness per layer
            layer.self_attn.q_proj.weight.mul_(s)
            layer.self_attn.k_proj.weight.mul_(s)
    return m, cfg


def make_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = set()
    for text in (pk.INTRO, pk.FILLER, pk.NEEDLE.format(key="0"), pk.QUESTION):
        words.update(text.replace(".", " . ").replace(",", " , ").replace("?", " ? ").split())
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    for d in "0123456789":
        vocab[d] = len(vocab)
    for w in sorted(words):
        if w not in vocab:
            vocab[w] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=True)])
    from tokenizers import decoders

    tk.decoder = decoders.WordPiece(prefix="##")  # joins with spaces
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", bos_token="[BOS]",
                                   eos_token="[EOS]", unk_token="[UNK]")


@pytest.fixture(scope="session")
def tiny_qwen():
    return make_tiny("qwen2")[0]


@pytest.fixture(scope="session")
def tiny_llama():
    return make_tiny("llama")[0]


@pytest.fixture(scope="session")
def tokenizer():
    return make_tokenizer()
