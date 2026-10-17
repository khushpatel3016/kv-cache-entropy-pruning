# Cross-Layer KV-Cache Pruning for Low-Latency LLM Inference on Resource-Constrained Hardware

**Khush Patel, Sneha Dash** · B.Tech CSE (AI/ML), VIT Chennai · BCSE306L Digital Assignment
Faculty guide: Dr. Vijayaprabakaran K

When a language model reads a long prompt it stores a key and value vector for every token in every layer (the **KV cache**). This cache grows with the prompt and often takes more GPU memory than the model itself. Pruning methods keep only part of it. They differ in **how they split a fixed memory budget across layers**: equally (SnapKV), by a fixed pyramid shape (PyramidKV), or adaptively per input (CAKE, DynamicKV).

This project asks a narrow, testable question: **how much does per-input adaptivity actually buy, and is attention entropy alone a good enough signal to drive it?** We study it on small (1.5B-3B) grouped-query-attention models on consumer GPUs.

## What is ours

| Component | What it does |
| --- | --- |
| **EntropyKV allocator** | Per layer, measures how spread out attention is (length-normalised entropy over a 32-token observation window) and gives spread-out layers a larger share of the budget, recomputed for every input. |
| **Static-vs-dynamic ablation** | The same allocator driven by a *frozen* entropy profile averaged offline over held-out prompts. Isolates the value of per-input recomputation (RQ2). |
| **Controlled comparison framework** | All methods share one prefill, one token-selection rule and an identical total budget, so differences come from allocation alone. Includes a random-allocation control. |

Related work we build on and compare with: H2O, StreamingLLM, SnapKV, PyramidKV, and **CAKE (Qin et al., ICLR 2025)**, which also uses attention entropy (combined with variance and cascading memory management). Our `cake_style` baseline reproduces CAKE's allocation rule inside our framework.

## Method

For every layer *l*, during prefill, only the last *w* = 32 queries are compared with all *n* keys (the full *n × n* attention matrix is never built):

```
A_l = softmax(q_window · K^T / sqrt(d))                       shape [heads, w, n]
H_l = mean over heads and window rows of  ( -Σ_j p_j log p_j ) / log(visible keys)    in [0, 1]
b_l = floor + (B - L·floor) · H_l^τ / Σ_k H_k^τ               (exact integer rounding, capped at n)
```

`τ = 0` reduces to uniform SnapKV; `τ > 1` sharpens the allocation. Inside each layer, tokens are kept by SnapKV's window score (attention mass from the window, average-pooled with kernel 7), and the window itself is always kept. Pruning happens once, after prefill; decoding then runs with each layer holding its own number of tokens.

## Quick start (any laptop, CPU only)

No NVIDIA GPU is needed for development, tests or the demo. Integrated graphics (e.g. Intel Iris Xe) are not used; everything runs on the CPU.

```bash
git clone https://github.com/khushpatel3016/kv-cache-entropy-pruning.git
cd kv-cache-entropy-pruning
python -m venv .venv                                 # Python 3.10-3.13
.venv\Scripts\activate                               # Windows  (Linux/Mac: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .

python -m pytest -q                                  # 57 tests, ~10 s
python scripts/demo.py                               # live demo, Qwen2.5-0.5B, 2K tokens, CPU
python -m kvprune.run --config configs/smoke_cpu.yaml
```

The smoke run should show `full ≥ snapkv ≥ random_tokens` on average. If not, stop and check the setup before running anything bigger. All reported experiments run on a GPU (Kaggle or Colab T4, see below); `configs/smoke_gpu.yaml` is the same check on a GPU.

## Reproducing every result

Main runs need a 16 GB GPU (free Kaggle or Colab T4). `notebooks/kaggle_run.ipynb` runs everything below across Kaggle's two T4s and resumes after session timeouts.

| Step | Command | Produces |
| --- | --- | --- |
| Static profile | `python scripts/calibrate_profile.py --model Qwen/Qwen2.5-3B-Instruct --out profiles/qwen2.5-3b-instruct.json` | frozen entropy profile (held-out tasks) |
| Main | `python -m kvprune.run --config configs/main_qwen3b.yaml --budgets 0.2` (then 0.1, 0.3, 0.5) | Fig 1, 2, 3, Table I |
| Passkey | `python -m kvprune.run --config configs/passkey_qwen3b.yaml` | Fig 4 |
| Ablations | `configs/ablation_qwen3b.yaml`, `ablation_window16_qwen3b.yaml`, `ablation_window64_qwen3b.yaml` | Fig 5 |
| Second model | `configs/llama3b_generalisation.yaml` (needs Llama-3.2 access) | Table II |
| Memory / latency | `python scripts/profile_memory.py --model Qwen/Qwen2.5-3B-Instruct --lengths 4096 8192 16384 --out results/profile_qwen3b.jsonl` | Fig 6 |
| Figures | `python scripts/make_figures.py` | `figures/*.png`, `*.pdf`, tables |

Every result row in `results/*.jsonl` records the method, budget, score, prediction, per-layer budgets, per-layer entropies, timings and environment. All figures regenerate from these files alone.

## Evaluation

* **LongBench** (6 tasks, one per category): qasper, hotpotqa, gov_report, trec, passage_retrieval_en, lcc. Official prompts, generation lengths, metrics and middle truncation (to 8K tokens).
* **Passkey retrieval**: 4K / 8K / 16K tokens × needle depth 0-100%.
* **Budgets**: 10, 20, 30, 50% of the full cache, identical across methods.
* **Statistics**: paired bootstrap (1,000 resamples), 95% intervals on every "ours minus baseline" difference (`figures/table_significance.csv`).

## Repository layout

```
src/kvprune/
  attention.py     custom attention backend: window statistics + uneven-length decoding
  allocation.py    uniform, pyramid, random, entropy and CAKE-style allocators
  selection.py     SnapKV, StreamingLLM and random token selection
  engine.py        prefill -> allocate -> prune -> greedy decode
  methods.py       method registry ("entropy:tau=2,floor=16" syntax)
  metrics.py       LongBench metrics
  stats.py         paired bootstrap
  run.py           resumable experiment runner
  data/            LongBench loader, passkey generator
scripts/           calibrate_profile, profile_memory, make_figures, demo
configs/           one YAML per experiment
tests/             57 tests (see below)
notebooks/         Kaggle notebook
```

## What the tests guarantee

* Captured scores and entropies equal those computed from the exact eager attention matrix.
* With the full cache, our decoding matches Hugging Face's reference logits (Llama and Qwen2).
* Decoding from per-layer caches of different lengths equals decoding from the full cache with dropped tokens masked out.
* Every allocator spends exactly the requested total and respects floors and caps.
* Budget 100% reproduces the full-cache output; equal entropies reproduce SnapKV token-for-token.
* The runner writes one row per job and resumes without duplicates.

## Limitations (stated in the report)

* Pruning happens after prefill, so **peak prefill memory is unchanged**; the savings are in the cache held during decoding and in decode latency. CAKE's cascading scheme addresses prefill peak; we leave it to future work.
* Batch size 1, greedy decoding.
* As in SnapKV, PyramidKV and CAKE, the question is inside the observation window during compression.
* One token-selection rule per layer (shared across a layer's heads), so the study isolates layer-level allocation.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `non-finite logits after prefill` | fp16 overflow. Use `dtype: bfloat16` on Ampere or newer GPUs, or `float32` for the 1.5B model. |
| CUDA out of memory | Lower `max_len` in the config, or add `load_in_4bit: true` (install `bitsandbytes`). |
| `401` / gated repo for Llama | Request access on the model page, then `huggingface-cli login`. |
| Windows symlink warning from huggingface_hub | Harmless; enable Developer Mode to silence it. |

## License

MIT. LongBench data is downloaded from its official Hugging Face repository at run time and is not redistributed here.
