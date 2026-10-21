"""Run one experiment stage across two GPUs (Kaggle 2x T4), safely inside a time limit.

    python scripts/kaggle_stage.py --stage main_20 --hours 11.2

Every job stops cleanly before the deadline (between samples, never mid-row),
so the notebook always reaches its final cell and saves its output. Rerunning
the same stage later continues exactly where it stopped.

Work is split so both GPUs finish at about the same time (see SPLITS).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PY = sys.executable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POLL = 300  # seconds between progress prints (tests lower this)
SHORT = ["qasper", "hotpotqa", "trec", "passage_retrieval_en", "lcc"]
# Balanced from timings measured on Kaggle T4s (Qwen2.5-3B, 8K context): gov_report writes
# 512-token summaries (~3.2 min/sample for 8 decodes) while the five other tasks together take
# ~2 h per 8-decode pass. Splitting gov_report by sample range makes both GPUs finish together.
SPLITS = {
    "one_budget": {0: ["gov_report[0:75]"], 1: ["gov_report[75:100]", *SHORT]},      # ~4 h each
    "three_budgets": {0: ["gov_report[0:67]"], 1: ["gov_report[67:100]", *SHORT]},   # ~9.5 h each
    "llama": {0: ["gov_report[0:35]"], 1: ["gov_report[35:50]", *SHORT]},             # ~1.5 h each
}

STAGES = {
    "main_20": "calibration + main experiment at 20% budget",
    "main_rest": "main experiment at 10%, 30%, 50%",
    "passkey_ablation": "passkey heatmaps (GPU 0) + ablations (GPU 1)",
    "window_profile": "window 16/64 ablations, memory/latency profiling, passkey at 2%/5%",
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


def wait(jobs, poll: int | None = None) -> list[int]:
    poll = POLL if poll is None else poll
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


def run_chains(chains: dict, clock: Clock) -> list[int]:
    """Run a list of steps per GPU, GPUs in parallel, steps in order. Each step is
    (kind, arg, tag) where kind is "cfg" (a kvprune.run config) or "cmd" (a command list).
    Every step receives the time left at the moment it starts, so the stage deadline holds."""
    import threading

    codes: list[int] = []
    lock = threading.Lock()

    def worker(gpu, steps):
        for kind, arg, tag in steps:
            left = clock.left()
            if left < 0.2:
                print(f"[skip] gpu{gpu} {tag}: not enough time left; rerun this stage", flush=True)
                with lock:
                    codes.append(0)
                continue
            if kind == "cfg":
                job = run_cfg(arg, gpu, clock, tag=tag)
            else:
                job = launch(arg, gpu, tag)
            with lock:
                codes.extend(wait([job]))

    threads = [threading.Thread(target=worker, args=(g, st)) for g, st in chains.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return codes


def main_split(cfg: str, budgets: list[str], prefix: str, clock: Clock, split: dict) -> list[int]:
    jobs = [run_cfg(cfg, g, clock, ["--budgets", *budgets, "--tasks", *tasks,
                                    "--output", f"results/{prefix}_gpu{g}.jsonl"], tag=f"{prefix}_gpu{g}")
            for g, tasks in split.items()]
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
        codes += main_split("configs/main_qwen3b.yaml", ["0.2"], "main_qwen3b", clock, SPLITS["one_budget"])
    elif a.stage == "main_rest":
        codes += main_split("configs/main_qwen3b.yaml", ["0.1", "0.3", "0.5"], "main_qwen3b", clock,
                            SPLITS["three_budgets"])
    elif a.stage == "passkey_ablation":
        codes += wait([run_cfg("configs/passkey_qwen3b.yaml", 0, clock),
                       run_cfg("configs/ablation_qwen3b.yaml", 1, clock)])
    elif a.stage == "window_profile":
        profile_cmd = [PY, "scripts/profile_memory.py", "--model", "Qwen/Qwen2.5-3B-Instruct",
                       "--lengths", "4096", "8192", "16384", "--out", "results/profile_qwen3b.jsonl"]
        codes += run_chains({
            0: [("cfg", "configs/ablation_window16_qwen3b.yaml", "ablation_window16_qwen3b"),
                ("cmd", profile_cmd, "profile")],
            1: [("cfg", "configs/ablation_window64_qwen3b.yaml", "ablation_window64_qwen3b"),
                ("cfg", "configs/passkey_extreme_qwen3b.yaml", "passkey_extreme_qwen3b")],
        }, clock)
    elif a.stage == "llama":
        codes += calibrate("meta-llama/Llama-3.2-3B-Instruct", "profiles/llama3.2-3b-instruct.json")
        codes += main_split("configs/llama3b_generalisation.yaml", ["0.2"], "llama3b_generalisation", clock,
                            SPLITS["llama"])

    print(f"stage {a.stage} finished in {(time.time() - t0) / 3600:.2f} h; exit codes {codes}", flush=True)
    if any(codes):
        print("SOME JOBS FAILED: see the log tails above. Finished rows are saved; fix and rerun.", flush=True)
    return int(any(codes))


if __name__ == "__main__":
    sys.exit(main())
