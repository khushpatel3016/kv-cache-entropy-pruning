"""LongBench (Bai et al., ACL 2024) loading and prompt construction.

Prompts and generation lengths follow the official LongBench configs
(config/dataset2prompt.json, config/dataset2maxlen.json). Over-long contexts are
truncated in the middle, keeping the first and last halves, as in the official
pred.py. Few-shot and code tasks are fed without a chat template, also as in
the official evaluation.

Data is read from the dataset's ``data.zip`` via huggingface_hub, which avoids
the deprecated dataset loading script entirely. If that fails we fall back to
``datasets.load_dataset(..., trust_remote_code=True)`` (requires datasets<4).
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass

LONGBENCH_REPO = "THUDM/LongBench"

PROMPTS = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "lcc": "Please complete the code given below. \n{context}Now, the code completion starts:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

MAX_NEW_TOKENS = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32, "2wikimqa": 32,
    "gov_report": 512, "multi_news": 512, "trec": 64, "samsum": 128, "passage_retrieval_en": 32,
    "passage_count": 32, "lcc": 64, "repobench-p": 64,
}

NO_CHAT_TEMPLATE = {"trec", "samsum", "lcc", "repobench-p"}

CATEGORY = {
    "narrativeqa": "single-doc QA", "qasper": "single-doc QA", "multifieldqa_en": "single-doc QA",
    "hotpotqa": "multi-doc QA", "2wikimqa": "multi-doc QA",
    "gov_report": "summarization", "multi_news": "summarization",
    "trec": "few-shot", "samsum": "few-shot",
    "passage_retrieval_en": "synthetic", "passage_count": "synthetic",
    "lcc": "code", "repobench-p": "code",
}

LONGBENCH_TASKS = list(PROMPTS)


@dataclass
class Sample:
    task: str
    idx: int
    prompt: str  # untruncated, template filled
    answers: list[str]
    all_classes: list[str] | None
    length: int  # LongBench's reported word count


def _cache_dir() -> str:
    d = os.environ.get("KVPRUNE_DATA", os.path.join(os.path.expanduser("~"), ".cache", "kvprune", "longbench"))
    os.makedirs(d, exist_ok=True)
    return d


def _load_jsonl_from_zip(task: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    out_dir = _cache_dir()
    jsonl = os.path.join(out_dir, "data", f"{task}.jsonl")
    if not os.path.exists(jsonl):
        zpath = hf_hub_download(repo_id=LONGBENCH_REPO, filename="data.zip", repo_type="dataset")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out_dir)
    with open(jsonl, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_rows(task: str) -> list[dict]:
    try:
        return _load_jsonl_from_zip(task)
    except Exception as zip_err:  # pragma: no cover - network dependent
        try:
            from datasets import load_dataset

            ds = load_dataset(LONGBENCH_REPO, task, split="test", trust_remote_code=True)
            return [dict(r) for r in ds]
        except Exception as ds_err:
            raise RuntimeError(f"could not load LongBench/{task}: {zip_err!r}; {ds_err!r}") from ds_err


def load_longbench(task: str, limit: int | None = None, offset: int = 0) -> list[Sample]:
    if task not in PROMPTS:
        raise KeyError(f"unknown LongBench task {task!r}")
    rows = _load_rows(task)
    rows = rows[offset: offset + limit if limit else None]
    out = []
    for i, r in enumerate(rows):
        prompt = PROMPTS[task].format(context=r["context"], input=r.get("input", ""))
        out.append(Sample(task, offset + i, prompt, list(r["answers"]), r.get("all_classes"), int(r.get("length", 0))))
    return out


def truncate_middle(token_ids: list[int], max_len: int) -> list[int]:
    if len(token_ids) <= max_len:
        return token_ids
    half = max_len // 2
    return token_ids[:half] + token_ids[len(token_ids) - (max_len - half):]


def build_longbench_prompt(sample: Sample, tokenizer, max_len: int) -> list[int]:
    """Truncate the prompt to ``max_len`` tokens (middle) and apply the chat template where LongBench does."""
    raw = tokenizer(sample.prompt, add_special_tokens=False).input_ids
    # Leave room for the chat template wrapper.
    budget = max_len - (64 if sample.task not in NO_CHAT_TEMPLATE else 0)
    text = tokenizer.decode(truncate_middle(raw, budget), skip_special_tokens=True)
    if sample.task not in NO_CHAT_TEMPLATE and getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return list(ids)
    return tokenizer(text, add_special_tokens=True).input_ids
