"""LongBench metrics, ported from the official metrics.py / eval.py.

Scores are in [0, 1] per sample (the paper reports x100). Each sample's score
is the max over its reference answers, as in the official evaluation.
"""

from __future__ import annotations

import re
import string
from collections import Counter

try:
    from rapidfuzz import fuzz as _fuzz

    def _ratio(a: str, b: str) -> float:
        return _fuzz.ratio(a, b) / 100.0
except ImportError:  # pragma: no cover
    import difflib

    def _ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred_tokens: list[str], gt_tokens: list[str]) -> float:
    common = Counter(pred_tokens) & Counter(gt_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p = same / len(pred_tokens)
    r = same / len(gt_tokens)
    return 2 * p * r / (p + r)


def qa_f1_score(prediction: str, ground_truth: str, **_) -> float:
    return _f1(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_score(prediction: str, ground_truth: str, **_) -> float:
    """ROUGE-L F1 over whitespace tokens (as the `rouge` package computes it)."""
    p, g = prediction.split(), ground_truth.split()
    lcs = _lcs(p, g)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)


def classification_score(prediction: str, ground_truth: str, all_classes=None, **_) -> float:
    matches = [c for c in (all_classes or []) if c in prediction]
    matches = [m for m in matches if not (m in ground_truth and m != ground_truth)]
    return 1.0 / len(matches) if ground_truth in matches else 0.0


def retrieval_score(prediction: str, ground_truth: str, **_) -> float:
    gt = re.findall(r"Paragraph (\d+)", ground_truth)
    if not gt:
        return 0.0
    nums = re.findall(r"\d+", prediction)
    return sum(n == gt[0] for n in nums) / len(nums) if nums else 0.0


def count_score(prediction: str, ground_truth: str, **_) -> float:
    nums = re.findall(r"\d+", prediction)
    return sum(n == str(ground_truth) for n in nums) / len(nums) if nums else 0.0


def code_sim_score(prediction: str, ground_truth: str, **_) -> float:
    line = ""
    for ln in prediction.lstrip("\n").split("\n"):
        if "`" not in ln and "#" not in ln and "//" not in ln:
            line = ln
            break
    return _ratio(line, ground_truth)


TASK_METRIC = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score,
    "gov_report": rouge_score, "multi_news": rouge_score, "samsum": rouge_score,
    "trec": classification_score,
    "passage_retrieval_en": retrieval_score, "passage_count": count_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}


def score_longbench(task: str, prediction: str, answers: list[str], all_classes=None) -> float:
    if task in ("trec", "samsum"):
        prediction = prediction.lstrip("\n").split("\n")[0]
    fn = TASK_METRIC[task]
    return max((fn(prediction, a, all_classes=all_classes) for a in answers), default=0.0)
