"""Run one experiment stage across two GPUs (Kaggle 2x T4), safely inside a time limit.

    python scripts/kaggle_stage.py --stage main_20 --hours 11.2

Every job stops cleanly before the deadline (between samples, never mid-row),
so the notebook always reaches its final cell and saves its output. Rerunning
the same stage later continues exactly where it stopped.

Work is split so both GPUs finish at about the same time: gov_report
(512-token summaries) costs about as much as the other five tasks together, so it
gets a GPU of its own.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PY = sys.executable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLIT = {0: ["gov_report"], 1: ["qasper", "hotpotqa", "trec", "passage_retrieval_en", "lcc"]}

STAGES = {
    "main_20": "calibration + main experiment at 20% budget",
    "main_rest": "main experiment at 10%, 30%, 50%",
    "passkey_ablation": "passkey heatmaps (GPU 0) + ablations (GPU 1)",
    "window_profile": "window 16 / 64 ablations + memory/latency profiling",
    "llama": "Llama-3.2-3B generalisation at 20%",
}


class Clock:
    def __init__(self, hours: float):
        self.deadline = time.time() + hours * 3600

    def left(self) -> float:
        return max(0.0, (self.deadline - time.time()) / 3600)


def _env(gpu: int) -> dict:
    return {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src"), "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": str(gpu)}


def launch(cmd: list[str], gpu: int, tag: str):
    log = os.path.join(ROOT, f"log_{tag}.txt")
    print(f"[start] gpu{gpu} {tag}: {' '.join(cmd[1:])}", flush=True)
    return subprocess.Popen(cmd, env=_env(gpu), cwd=ROOT, stdout=open(log, "a"), stderr=subprocess.STDOUT), log


def wait(jobs, poll: int = 300) -> list[int]:
    procs = [p for p, _ in jobs]
    while any(p.poll() is None for p in procs):
        time.sleep(poll)
        for _, log in jobs:
            try:
                last = open(log).read().strip().splitlines()[-1]
                print(f"{os.path.basename(log)} | {last[:160]}", flush=True)
            except (OSError, IndexError):
                pass
    codes = [p.returncode for p in procs]
    for _, log in jobs:
        print(f"---- tail of {os.path.basename(log)}", flush=True)
        try:
            print("".join(open(log).readlines()[-12:]), flush=True)
        except OSError:
            pass
    return codes


def run_cfg(cfg: str, gpu: int, clock: Clock, extra=(), tag: str | None = None):
    tag = tag or os.path.basename(cfg).replace(".yaml", "")
    cmd = [PY, "-m", "kvprune.run", "--config", cfg, "--max-hours", f"{clock.left():.3f}", *extra]
    return launch(cmd, gpu, tag)


def main_split(cfg: str, budgets: list[str], prefix: str, clock: Clock) -> list[int]:
    jobs = [run_cfg(cfg, g, clock, ["--budgets", *budgets, "--tasks", *tasks,
                                    "--output", f"results/{prefix}_gpu{g}.jsonl"], tag=f"{prefix}_gpu{g}")
            for g, tasks in SPLIT.items()]
    return wait(jobs)


def calibrate(model: str, out: str) -> list[int]:
    if os.path.exists(os.path.join(ROOT, out)):
        print("profile exists:", out, flush=True)
        return [0]
    return wait([launch([PY, "scripts/calibrate_profile.py", "--model", model, "--out", out], 0, "calibrate")])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--hours", type=float, default=11.2, help="hard time budget for the whole stage")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    os.makedirs("results", exist_ok=True)
    clock = Clock(a.hours)
    t0 = time.time()
    print(f"stage {a.stage}: {STAGES[a.stage]} (budget {a.hours:.1f} h)", flush=True)

    codes: list[int] = []
    if a.stage == "main_20":
        codes += calibrate("Qwen/Qwen2.5-3B-Instruct", "profiles/qwen2.5-3b-instruct.json")
        codes += main_split("configs/main_qwen3b.yaml", ["0.2"], "main_qwen3b", clock)
    elif a.stage == "main_rest":
        codes += main_split("configs/main_qwen3b.yaml", ["0.1", "0.3", "0.5"], "main_qwen3b", clock)
    elif a.stage == "passkey_ablation":
        codes += wait([run_cfg("configs/passkey_qwen3b.yaml", 0, clock),
                       run_cfg("configs/ablation_qwen3b.yaml", 1, clock)])
    elif a.stage == "window_profile":
        codes += wait([run_cfg("configs/ablation_window16_qwen3b.yaml", 0, clock),
                       run_cfg("configs/ablation_window64_qwen3b.yaml", 1, clock)])
        if clock.left() > 0.75:
            codes += wait([launch([PY, "scripts/profile_memory.py", "--model", "Qwen/Qwen2.5-3B-Instruct",
                                   "--lengths", "4096", "8192", "16384",
                                   "--out", "results/profile_qwen3b.jsonl"], 0, "profile")])
        else:
            print("not enough time left for profiling; rerun this stage", flush=True)
    elif a.stage == "llama":
        codes += calibrate("meta-llama/Llama-3.2-3B-Instruct", "profiles/llama3.2-3b-instruct.json")
        codes += main_split("configs/llama3b_generalisation.yaml", ["0.2"], "llama3b_generalisation", clock)

    print(f"stage {a.stage} finished in {(time.time() - t0) / 3600:.2f} h; exit codes {codes}", flush=True)
    if any(codes):
        print("SOME JOBS FAILED: see the log tails above. Finished rows are saved; fix and rerun.", flush=True)
    return int(any(codes))


if __name__ == "__main__":
    sys.exit(main())
