import json
import os

import numpy as np
import pytest
import torch
import yaml

from conftest import make_tiny, make_tokenizer
from kvprune.data.longbench import truncate_middle
from kvprune.data.passkey import make_passkey_prompt, passkey_correct
from kvprune.metrics import score_longbench
from kvprune.stats import paired_bootstrap


def test_qa_f1():
    assert score_longbench("hotpotqa", "The Eiffel Tower", ["eiffel tower"]) == 1.0
    assert score_longbench("hotpotqa", "Paris", ["London"]) == 0.0
    assert 0 < score_longbench("qasper", "red apple pie", ["apple pie"]) < 1


def test_retrieval_classification_code():
    assert score_longbench("passage_retrieval_en", "Paragraph 12", ["Paragraph 12"]) == 1.0
    assert score_longbench("passage_retrieval_en", "Paragraph 3 or 12", ["Paragraph 12"]) == 0.5
    classes = ["Location", "Abbreviation", "Abbreviation of entity"]
    assert score_longbench("trec", "Location\nmore", ["Location"], classes) == 1.0
    assert score_longbench("lcc", "x = 1\n", ["x = 1"]) == 1.0


def test_rouge():
    assert score_longbench("gov_report", "a b c d", ["a b c d"]) == 1.0
    assert score_longbench("gov_report", "a b", ["c d"]) == 0.0


def test_truncate_middle():
    x = list(range(100))
    t = truncate_middle(x, 10)
    assert t == [0, 1, 2, 3, 4, 95, 96, 97, 98, 99]
    assert truncate_middle(x, 200) == x


@pytest.mark.parametrize("depth", [0.0, 0.5, 1.0])
def test_passkey_prompt(depth):
    tok = make_tokenizer()
    ids, key = make_passkey_prompt(tok, 600, depth, seed=3)
    assert 450 < len(ids) <= 650
    text = tok.decode(ids)
    assert " ".join(key) in text  # digits tokenised individually
    pos = text.find("pass key is")
    assert pos >= 0
    assert passkey_correct(f"{key}.", key) == 1.0
    assert passkey_correct("12345", key) == float(key == "12345")


def test_bootstrap():
    a = np.ones(50)
    b = np.zeros(50)
    d, lo, hi, p = paired_bootstrap(a, b)
    assert d == 1.0 and lo == 1.0 and hi == 1.0 and p == 0.0
    rng = np.random.default_rng(0)
    x = rng.random(200)
    d, lo, hi, p = paired_bootstrap(x, x + 0.0)
    assert d == 0.0


def test_runner_end_to_end(tmp_path):
    """Full pipeline on a saved tiny model: passkey jobs, all methods, resume."""
    from kvprune.run import run

    m, cfg = make_tiny("qwen2", layers=4)
    tok = make_tokenizer()
    mdir = tmp_path / "tiny"
    m.config.vocab_size = 128
    m.save_pretrained(mdir)
    tok.save_pretrained(mdir)
    prof = tmp_path / "prof.json"
    prof.write_text(json.dumps({"entropies": [0.3, 0.5, 0.7, 0.9]}))
    out = tmp_path / "res.jsonl"
    conf = {
        "name": "t", "model": str(mdir), "device": "cpu", "window": 16,
        "methods": ["full", "streaming", "snapkv", "pyramidkv", "random_alloc", "entropy_static", "entropy", "cake_style"],
        "budgets": [0.2, 0.5], "static_profile": str(prof),
        "passkey": {"lengths": [300], "depths": [0.0, 1.0], "trials": 2},
        "output": str(out),
    }
    run(conf)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 4 * (1 + 7 * 2)
    assert {r["method"] for r in rows} >= {"full", "entropy", "snapkv"}
    for r in rows:
        assert sum(r["layer_budgets"]) == (r["n_tokens"] * 4 if r["method"] == "full" else int(r["budget"] * r["n_tokens"] * 4))
    run(conf)  # resume: nothing new
    assert len(out.read_text().splitlines()) == len(rows)


def _tiny_run_conf(tmp_path, out_name):
    m, cfg = make_tiny("qwen2", layers=4)
    tok = make_tokenizer()
    mdir = tmp_path / "tiny"
    if not mdir.exists():
        m.save_pretrained(mdir)
        tok.save_pretrained(mdir)
    return {
        "name": "t2", "model": str(mdir), "device": "cpu", "window": 16,
        "methods": ["full", "snapkv", "entropy"], "budgets": [0.2],
        "passkey": {"lengths": [300], "depths": [0.0, 1.0], "trials": 1},
        "output": str(tmp_path / out_name),
    }


def test_runner_time_limit_stops_cleanly(tmp_path):
    from kvprune.run import run

    conf = _tiny_run_conf(tmp_path, "a.jsonl")
    conf["max_hours"] = 0.0
    run(conf)
    assert not (tmp_path / "a.jsonl").exists() or (tmp_path / "a.jsonl").read_text() == ""
    del conf["max_hours"]
    run(conf)
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == 2 * 3


def test_runner_resume_across_files(tmp_path):
    """Rows finished in one output file are not repeated when the same experiment writes to another file."""
    from kvprune.run import run

    conf = _tiny_run_conf(tmp_path, "gpu0.jsonl")
    run(conf)
    n0 = len((tmp_path / "gpu0.jsonl").read_text().splitlines())
    conf["output"] = str(tmp_path / "gpu1.jsonl")
    conf["budgets"] = [0.2, 0.5]  # only the 0.5 jobs are new; 'full' must not repeat
    run(conf)
    rows = [json.loads(l) for l in (tmp_path / "gpu1.jsonl").read_text().splitlines()]
    assert n0 == 2 * 3
    assert len(rows) == 2 * 2
    assert all(r["budget"] == 0.5 for r in rows)


def test_task_slice_parsing():
    from kvprune.run import parse_task_slice as f

    assert f("gov_report") == ("gov_report", None, None)
    assert f("gov_report[0:67]") == ("gov_report", 0, 67)
    assert f("lcc[50:]") == ("lcc", 50, None)


def test_kaggle_splits_cover_every_sample_once():
    """Each stage split must evaluate every (task, sample) exactly once across the two GPUs."""
    import importlib.util

    from kvprune.data.longbench import LONGBENCH_TASKS
    from kvprune.run import parse_task_slice

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "kaggle_stage.py")
    spec = importlib.util.spec_from_file_location("kaggle_stage", path)
    ks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ks)
    limits = {"one_budget": 100, "three_budgets": 100, "llama": 50}
    tasks = ["qasper", "hotpotqa", "gov_report", "trec", "passage_retrieval_en", "lcc"]
    for name, split in ks.SPLITS.items():
        n = limits[name]
        seen = {t: [0] * n for t in tasks}
        for gpu_tasks in split.values():
            for spec_ in gpu_tasks:
                t, lo, hi = parse_task_slice(spec_)
                assert t in LONGBENCH_TASKS
                for i in list(range(n))[lo:hi]:
                    seen[t][i] += 1
        for t in tasks:
            assert seen[t] == [1] * n, (name, t)
