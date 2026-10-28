Cross-platform speed, Qwen2.5-0.5B fp32, 4052 tokens, EntropyKV at 10%

| platform   | hardware                                                  |   length |   prefill_ms |   kv_full_mb |   kv_pruned_mb |   decode_full_ms |   decode_ours_ms |   decode_snapkv_ms |   speedup |   overhead_ms |   overhead_pct_prefill |
|:-----------|:----------------------------------------------------------|---------:|-------------:|-------------:|---------------:|-----------------:|-----------------:|-------------------:|----------:|--------------:|-----------------------:|
| NVIDIA T4  | Tesla T4                                                  |     4052 |      1537.03 |        94.97 |            9.5 |            36.11 |            27.99 |              28.36 |      1.29 |          9.93 |                   0.65 |
| Laptop CPU | cpu (Intel64 Family 6 Model 186 Stepping 3, GenuineIntel) |     4052 |     28950.8  |        94.97 |            9.5 |           172.24 |           113.19 |             100.87 |      1.52 |         30.95 |                   0.11 |


Passkey accuracy (%), same prompts on every machine (2K and 4K tokens, 5 depths x 5 trials):

|                      |   NVIDIA T4 |   Laptop CPU |
|:---------------------|------------:|-------------:|
| ('cake_style', 0.05) |         100 |          100 |
| ('cake_style', 0.1)  |         100 |          100 |
| ('cake_style', 0.2)  |         100 |          100 |
| ('entropy', 0.05)    |          20 |           20 |
| ('entropy', 0.1)     |          60 |           60 |
| ('entropy', 0.2)     |         100 |          100 |
| ('full', 1.0)        |         100 |          100 |
| ('pyramidkv', 0.05)  |          20 |           20 |
| ('pyramidkv', 0.1)   |          56 |           56 |
| ('pyramidkv', 0.2)   |          98 |           98 |
| ('snapkv', 0.05)     |          50 |           50 |
| ('snapkv', 0.1)      |         100 |          100 |
| ('snapkv', 0.2)      |         100 |          100 |

Identical answers to the T4 on Laptop CPU: 100.0% of 650 runs
