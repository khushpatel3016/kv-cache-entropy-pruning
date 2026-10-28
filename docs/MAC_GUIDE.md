# On-device study on a MacBook (Apple Silicon)

This guide runs the project on an Apple Silicon MacBook using the Mac's built-in GPU
(PyTorch "MPS" backend). It measures what pruning saves on a laptop (KV-cache memory
and decode speed) and checks passkey accuracy for full cache, SnapKV, PyramidKV,
CAKE-style and EntropyKV with Qwen2.5-0.5B.

Total time: about 1-1.5 hours, most of it unattended. Keep the Mac plugged in.

## 1. One-time setup (10 min)

1. Accept Khush's GitHub collaborator invite (check your email or github.com/notifications).
2. Open **Terminal** and check Python (3.10-3.13 is fine):
   ```bash
   python3 --version
   ```
   If it is missing, install it from python.org or with `brew install python@3.12`.
3. Get the code and tell git who you are (use the email of your GitHub account):
   ```bash
   cd ~/Documents
   git clone https://github.com/khushpatel3016/kv-cache-entropy-pruning.git
   cd kv-cache-entropy-pruning
   git config user.name "Sneha Dash"
   git config user.email "YOUR_GITHUB_EMAIL"
   ```
4. Create the environment and install:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install torch
   pip install -r requirements.txt
   pip install -e .
   ```
5. Check that the Mac GPU is visible. This must print `True`:
   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```
6. Run the tests. Expected: `63 passed`.
   ```bash
   python -m pytest -q
   ```

Every time you open a new Terminal later: `cd ~/Documents/kv-cache-entropy-pruning && source .venv/bin/activate`.

## 2. Quick demo (2 min)

```bash
python scripts/demo.py --device mps
```
The first run downloads Qwen2.5-0.5B (about 1 GB). It prints each method's answer to a
hidden-passkey question, whether it was correct, and the KV-cache size, and saves
`figures/demo_budgets.png`. This is also the demo for the viva.

## 3. Memory and speed (about 15-25 min)

Close other heavy apps (browser tabs with video, games) so timings are clean, then:
```bash
python scripts/profile_memory.py --model Qwen/Qwen2.5-0.5B-Instruct --device mps \
    --lengths 2048 4096 --methods full snapkv pyramidkv entropy cake_style \
    --budgets 0.1 0.2 0.5 --repeats 3 --out results/profile_mac_qwen05b.jsonl
```
Each line it prints is one measurement: KV size (MB), decode speed (ms per token) and
the time taken to choose budgets and tokens.

## 4. Passkey accuracy (about 20-40 min)

```bash
python -m kvprune.run --config configs/passkey_mac_qwen05b.yaml
```
If it is interrupted (lid closed, battery), run the same command again: it continues
where it stopped.

## 5. Figures

```bash
python scripts/make_figures.py
```
Your results appear as `figures/fig10_mac_memory_latency.png` and `figures/table_mac.md`
(the other figures are the Kaggle T4 study and stay unchanged).

## 6. Commit and push your results

```bash
git add results/profile_mac_qwen05b.jsonl results/passkey_mac_qwen05b.jsonl figures/
git commit -m "On-device study on MacBook Air (Apple Silicon, MPS): memory, speed, passkey"
git pull --rebase
git push
```
Then send Khush the contents of `figures/table_mac.md` so it can go into the report.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `... is not currently implemented for the MPS device` | Run `export PYTORCH_ENABLE_MPS_FALLBACK=1` and run the same command again. |
| `MPS backend out of memory` | Use `--lengths 2048` only in step 3, and in `configs/passkey_mac_qwen05b.yaml` set `lengths: [2048]`. |
| `non-finite logits after prefill` | Add `--dtype float32` (step 3) or set `dtype: float32` in the config. |
| `pip install` fails on a package | Run `pip install --upgrade pip` and try again; make sure `(.venv)` is shown in the prompt. |
| `git push` is rejected | Run `git pull --rebase` then `git push` again. |
