# Sneha's part: Apple Silicon study + prototype (MacBook)

The final IEEE report compares the same method on three machines:

| Machine | Who | What runs |
| --- | --- | --- |
| NVIDIA T4 (Kaggle) | Khush | main study (Qwen2.5-3B, Llama-3.2-3B) + the cross-platform test |
| **Apple Silicon MacBook (MPS)** | **Sneha** | **the cross-platform test** + her own prototype |
| Intel laptop CPU | Khush | the cross-platform test |

The **cross-platform test** is identical on every machine (Qwen2.5-0.5B, float32, 2K and 4K
tokens, same budgets, same prompts). Only the hardware changes, so the report can compare
speed, memory and accuracy fairly. **Do not change any setting in the commands below**,
or the comparison stops being fair.

Total time: about 3 hours, half of it waiting. Keep the Mac plugged in.

---

## Task 1. Setup (15 min)

1. Accept Khush's GitHub invite (email or github.com/notifications).
2. In **Terminal**:
   ```bash
   python3 --version            # 3.10-3.13 is fine; else: brew install python@3.12
   cd ~/Documents
   git clone https://github.com/khushpatel3016/kv-cache-entropy-pruning.git
   cd kv-cache-entropy-pruning
   git config user.name "Sneha Dash"
   git config user.email "THE_EMAIL_ON_YOUR_GITHUB_ACCOUNT"
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install torch
   pip install -r requirements.txt
   pip install -e .
   python -c "import torch; print(torch.backends.mps.is_available())"   # must print True
   python -m pytest -q                                                    # must say 63 passed
   ```
   The email must be one that is listed in **your** GitHub account (Settings -> Emails),
   otherwise your commits will not show your name.

Every new Terminal later: `cd ~/Documents/kv-cache-entropy-pruning && source .venv/bin/activate`.

## Task 2. Add your existing prototype to the repo (10 min)

Your `KVCacheProject` goes in its own folder, without Mac junk files:
```bash
mkdir -p sneha_mac_prototype
rsync -a --exclude='__MACOSX' --exclude='.venv' --exclude='__pycache__' --exclude='.DS_Store' \
    ~/PATH/TO/KVCacheProject/ sneha_mac_prototype/
mv sneha_mac_prototype/results sneha_mac_prototype/results_v1
git add sneha_mac_prototype
git commit -m "Add independent Mac prototype (v1): entropy profile and passkey benchmark"
git push
```
(Replace `~/PATH/TO/KVCacheProject` with the real location.) `results_v1` keeps your first
results exactly as they were; they are part of the story (see Task 4).

## Task 3. Cross-platform test on the Mac (about 1-1.5 h, mostly waiting)

Close heavy apps (video, games, many browser tabs) so timings are clean.

**3a. Demo (2 min).** Also the viva demo.
```bash
python scripts/demo.py --device mps
```

**3b. Memory and speed (20-40 min).** Copy exactly:
```bash
python scripts/profile_memory.py --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype float32 \
    --lengths 2048 4096 --methods full snapkv pyramidkv entropy cake_style \
    --budgets 0.1 0.2 0.5 --repeats 3 --out results/profile_mac_qwen05b.jsonl
```

**3c. Passkey accuracy (20-40 min).** If it stops (lid closed, battery), run it again; it resumes.
```bash
python -m kvprune.run --config configs/passkey_mac_qwen05b.yaml
```

**3d. Push your results.**
```bash
git add results/profile_mac_qwen05b.jsonl results/passkey_mac_qwen05b.jsonl
git commit -m "Cross-platform test on MacBook (Apple Silicon, MPS): memory, speed, passkey"
git pull --rebase
git push
```

## Task 4. Fix and rerun your prototype (v2) (about 1 h)

Your v1 prototype gave **0% accuracy for every pruned policy, even at 50%**. The full
cache got 100%, and the main pipeline still finds the key at 5%, so the cause is in the
prototype's design, not in pruning itself. Likely causes, visible in the code:

1. **The question gets deleted.** Tokens are ranked by attention averaged over *all* query
   positions, which favours early tokens (every later token can see them). The question at
   the end ("What is the important passkey? ... Answer:") scores low and is pruned; only the
   very last token is protected.
2. **Unequal budgets.** `entropy` keeps about 44% while `fixed`/`random` keep 50%, so the
   comparison is not at the same memory.
3. **Very short prompts** (about 200 tokens), because the full attention matrix is built.

Make these changes in `sneha_mac_prototype/` only (do not edit the main `src/kvprune`):

| Change | Where |
| --- | --- |
| Score tokens using only the **last 32 query rows** (the observation window): `attention[..., -32:, :]` averaged over heads and those rows | `src/entropy_calculator.py` |
| Always keep the **first 4 tokens** (attention sinks) and the **last 32 tokens** (the question), then fill the rest by score | `src/token_selector.py` |
| Give every pruned policy the **same number of kept tokens**; entropy only changes *which* tokens (via its layer weighting), not *how many* | `experiments/passkey_benchmark.py` |
| Longer prompts: about **1,000 tokens** (more filler sentences), `max_input_tokens: 1024`, and make sure the passkey is never truncated | `make_prompt`, `configs/config.json` |
| Run **budgets 50%, 30%, 20%, 10%**, **30 trials** each, and save to `results_v2/` | `experiments/passkey_benchmark.py` |
| Print the generated text for 2 trials per policy, to see what the model answers | `experiments/passkey_benchmark.py` |

Then:
```bash
cd sneha_mac_prototype
python experiments/passkey_benchmark.py      # writes results_v2/
cd ..
git add sneha_mac_prototype
git commit -m "Prototype v2: observation-window scoring, keep question tokens, matched budgets"
git push
```

What to send Khush: a small table **v1 vs v2** (accuracy per policy per budget), and one
sentence on which change made the biggest difference. Whatever the numbers are, report
them as they are.

## Task 5. Viva preparation

Be ready to explain in your own words:
1. What the KV cache is and why it grows with the prompt.
2. What "pruning with a budget" means (keep 10% of the cached tokens, drop the rest).
3. Why we compare machines with the **same** model, precision and prompts.
4. Your Mac numbers: decode speed full vs 10%, memory saved, overhead.
5. Why your v1 prototype failed and what v2 changed.
6. Run the demo (`python scripts/demo.py --device mps`) and explain its output.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `... not currently implemented for the MPS device` | `export PYTORCH_ENABLE_MPS_FALLBACK=1`, then rerun. |
| `MPS backend out of memory` | Close other apps and rerun. If it repeats, tell Khush (do not change the settings yourself). |
| `pytest` does not say 63 passed | Check `(.venv)` is in the prompt; rerun `pip install -e .` |
| `git push` rejected | `git pull --rebase`, then `git push`. |
| Commits show the wrong name on GitHub | `git config user.email` must be an email on your GitHub account. |
