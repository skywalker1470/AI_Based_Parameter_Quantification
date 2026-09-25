# Ablation study: FD001

Seeds: [0, 1, 2, 3, 4] (mean ± std over seeds). Epochs: 40. Test = last window of each of the 100 test engines; RMSE against RUL clipped at 125 unless stated. Fault = RUL ≤ 20; alarm = predicted RUL ≤ 20.

## 1. Leakage: engine-level vs. window-level split

| Split | Val RMSE | Test RMSE | Test − Val (optimism gap) |
|---|---|---|---|
| engine-level split (default) | 14.65 ± 0.64 | 14.25 ± 0.42 | -0.39 ± 0.69 |
| window-level split (leaky) | 15.13 ± 0.15 | 14.20 ± 0.48 | -0.93 ± 0.58 |

## 2. Preprocessing: window size and RUL clipping

| Variant (MLP) | Test RMSE (clipped labels) | Test RMSE (true RUL) | PHM08 score | Params |
|---|---|---|---|---|
| window 1 | 17.69 ± 0.20 | 18.74 ± 0.17 | 1059 ± 109 | 10625 |
| window 10 | 18.27 ± 0.30 | 19.18 ± 0.24 | 1126 ± 175 | 30209 |
| window 30 (default) | 14.25 ± 0.42 | 15.36 ± 0.52 | 328 ± 34 | 73729 |
| window 50 | 16.63 ± 0.60 | 17.47 ± 0.71 | 1239 ± 1190 | 117249 |
| window 30, no RUL clipping | 24.96 ± 0.99 | 24.73 ± 1.01 | 26866 ± 18869 | 73729 |

## 3. Models vs. baselines

| Approach | Test RMSE | NMSE | PHM08 score | Fault F1 (RUL ≤ 20) | Train time (s) | Latency, batch 1 (µs) | Params |
|---|---|---|---|---|---|---|---|
| mean_predictor | 40.48 | 1.020 | 17604 | 0.000 | 0 (fit only) | 0 | 1 |
| hi_trend | 37.19 | 0.861 | 18837 | 0.400 | 0 (fit only) | 297 | 16 |
| hi_similarity | 15.02 | 0.140 | 428 | 0.828 | 0 (fit only) | 2633 | 16 |
| mlp | 14.25 ± 0.42 | 0.127 ± 0.008 | 328 ± 34 | 0.802 ± 0.028 | 6.1 ± 0.3 | 75 ± 41 | 73729 |
| bilstm | 14.34 ± 0.59 | 0.128 ± 0.011 | 375 ± 68 | 0.823 ± 0.038 | 4.3 ± 0.8 | 370 ± 192 | 46657 |
