"""Synthetic passkey retrieval (Mohtashami & Jaggi, 2023; used by Quest, Infini-attention).

A random 5-digit key is hidden at a controlled relative depth inside
repetitive filler text of a controlled token length. The model must repeat
it. Because the key is the only informative content, any drop in accuracy
after pruning means the pruning threw the key away.
"""

from __future__ import annotations

import random
import re

INTRO = ("There is an important piece of information hidden inside a lot of irrelevant text. "
         "Find it and memorize it. I will quiz you about the important information there.\n\n")
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
NEEDLE = "The pass key is {key}. Remember it. {key} is the pass key. "
QUESTION = "\n\nWhat is the pass key? The pass key is"


def make_passkey_prompt(tokenizer, target_tokens: int, depth: float, seed: int = 0) -> tuple[list[int], str]:
    """Return (token ids, key) for a prompt of about ``target_tokens`` tokens.

    ``depth`` in [0, 1] is where the needle sits within the filler
    (0 = right after the intro, 1 = just before the question).
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be in [0, 1]")
    key = str(random.Random(seed).randint(10000, 99999))
    needle = NEEDLE.format(key=key)

    def n_tok(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False).input_ids)

    fixed = n_tok(INTRO) + n_tok(needle) + n_tok(QUESTION) + 8
    per = max(1, n_tok(FILLER * 10) / 10)
    reps = max(1, int((target_tokens - fixed) / per))
    before = int(round(reps * depth))
    text = INTRO + FILLER * before + needle + FILLER * (reps - before) + QUESTION
    ids = tokenizer(text, add_special_tokens=True).input_ids
    return ids, key


def passkey_correct(prediction: str, key: str) -> float:
    m = re.search(r"\d{5}", prediction)
    return float(m is not None and m.group(0) == key)
