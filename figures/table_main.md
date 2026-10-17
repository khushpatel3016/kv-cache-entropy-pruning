LongBench scores (x100) at 20% KV budget; full cache is the upper bound.

|                   |   GovReport |   HotpotQA |   LCC |   PassageRet. |   Qasper |   TREC |   Average |
|:------------------|------------:|-----------:|------:|--------------:|---------:|-------:|----------:|
| Full cache        |       18.57 |      44.52 | 29.52 |            39 |    34.49 |     65 |     38.52 |
| EntropyKV (ours)  |       16.62 |      45.31 | 25.94 |            39 |    30.79 |     64 |     36.94 |
| EntropyKV-static  |       16.98 |      45.31 | 27.05 |            39 |    30.49 |     63 |     36.97 |
| CAKE-style        |       17.05 |      45.18 | 26.87 |            38 |    31.09 |     63 |     36.87 |
| PyramidKV         |       15.56 |      45.51 | 22.35 |            40 |    26.96 |     59 |     34.9  |
| SnapKV            |       16.59 |      45.64 | 26.76 |            39 |    32.33 |     64 |     37.39 |
| StreamingLLM      |       15.63 |      36.98 | 22.9  |            12 |    20.17 |     52 |     26.61 |
| Random allocation |       15.84 |      43.92 | 25.18 |            22 |    28.3  |     56 |     31.87 |
