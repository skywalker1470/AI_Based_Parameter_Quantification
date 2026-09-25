
# AI-Based Fault Detection & Parameter Quantification on Engine Sensor Data

Portfolio project demonstrating surrogate modeling + fault/anomaly detection on
multivariate sensor time-series from a mechanical system, built as preparation
for AI-based fault detection roles on internal combustion engines.

📄 **Report:** [report/report.pdf](report/report.pdf), a short write-up of
the method, results and ablations (LaTeX source:
[report/report.tex](report/report.tex)).

## Motivation

Physics-based / exact estimation of an engine's health state (remaining
useful life, degradation level) from raw sensor streams is often expensive at
inference time (per-unit model fitting, iterative solvers, high-fidelity
simulation). A learned **surrogate model** can approximate that expensive
computation with a single amortized forward pass, at the cost of some
accuracy. This project quantifies that accuracy/cost trade-off against
classic non-deep-learning prognostics baselines, layers a fault-detection
mechanism on top of the surrogate's output, and checks the main design
choices with a multi-seed ablation study.

## Dataset

[NASA C-MAPSS Turbofan Engine Degradation Simulation](CMAPSSData/readme.txt)
(FD001 subset used by default: 100 train + 100 test engines, one operating
condition, one fault mode). Each row is one operational cycle of one engine:

```
unit, cycle, op_setting_1..3, sensor_1..21
```

Training trajectories run to failure; test trajectories are truncated, with
true Remaining Useful Life (RUL) at truncation given in `RUL_FD00x.txt`. RUL
is the health indicator / operating parameter this project quantifies, and
"fault" is defined as RUL falling below a threshold (near end-of-life).

## Method

```
src/
  data.py        - load C-MAPSS files, attach RUL labels, drop constant sensors
  windows.py      - sliding-window segmentation, unit-level (leakage-free) train/val split
  surrogate.py    - MLP and BiLSTM surrogate models (PyTorch), training loop
  anomaly.py      - Isolation Forest + rule-based fault detectors, validation threshold tuning, k-of-n voting
  baseline.py     - non-DL prognostics baselines on a learned health index (HI):
                     HI trend extrapolation and similarity-based RUL
  evaluate.py     - RMSE, NMSE, PHM08 asymmetric score, precision/recall/F1, bootstrap CIs
  plots.py        - training curves, predicted-vs-true RUL, accuracy-vs-cost plot
  pipeline.py     - end-to-end script wiring the above together
  ablation.py     - multi-seed ablation study (leakage, preprocessing, models vs. baselines)
notebooks/
  01_pipeline_walkthrough.ipynb - runs the pipeline and inspects results interactively
```

**Pipeline stages:**

1. **Data prep**: load raw sensor streams, drop sensors with ~zero variance
   (uninformative in the single-operating-condition subsets), z-score
   normalize using train statistics only, clip RUL labels at 125 cycles
   (standard C-MAPSS practice, since degradation is negligible early in life).
   Sliding windows (default: 30 cycles, stride 1) are built **per engine**,
   and train/validation split is done **by whole engine**, never by cutting
   a single run, so no cycle from a validation engine's trajectory leaks
   into training.

2. **Baselines (no deep learning)**. C-MAPSS ships no physics model, so
   instead of claiming a "physics" baseline, two classic data-driven
   prognostics methods are used, both on a **health index** (HI): a linear
   combination of the sensors fit so HI ≈ 1 in each training engine's first
   30 cycles and ≈ 0 in its last 30.
   - *HI trend extrapolation*: fit a line to the test engine's last 50 HI
     values and extrapolate to the population failure level.
   - *HI similarity*: slide the test engine's HI trajectory along every
     run-to-failure training trajectory, take the best match in each, and
     use the median remaining life of the 10 closest matches (the approach
     family behind the PHM08 challenge's winning method). No training, but
     every prediction scans the whole training library.

3. **Surrogate models**: a lightweight MLP (flattened window → RUL) and a
   small BiLSTM (sequence-aware). Both are trained with MSE, Adam and early
   stopping on the validation engines, with all seeds fixed.

4. **Fault/anomaly detection layer**. Each detector outputs a score, and its
   alarm threshold is **tuned on the validation engines** (max F1), never on
   the test set:
   - **Surrogate threshold**: alarm when predicted RUL is low.
   - **Isolation Forest** on per-sensor summary features (recent level +
     trend slope), fit on early-life windows only (RUL at the 125 plateau).
   - **Rule-based detector**: number of sensors deviating > 3σ from the
     early-life baseline, a simple analogue of FP-Growth-style
     "co-occurring deviation" rules.
   - **Combiners**: OR, AND, 2-of-3 vote, and a stacked logistic regression
     on the three scores.

5. **Evaluation**: RMSE, NMSE and the PHM08 asymmetric score (penalizes late
   predictions more than early ones) for RUL; precision/recall/F1 for
   detection, with bootstrap 95% CIs, on both the last window per test
   engine and on every test window; batch-size-1 latency for cost.

6. **Ablation study** (`src/ablation.py`, 5 seeds, mean ± std):
   - *Leakage*: engine-level vs. window-level train/val split.
   - *Preprocessing*: window size 1 / 10 / 30 / 50, RUL clipping on/off.
   - *Models vs. baselines*: mean predictor, HI trend, HI similarity, MLP,
     BiLSTM, on accuracy and cost (training time, latency, parameters).

## Running it

```bash
# Linux / WSL (on Ubuntu first: sudo apt install python3-full python3-venv)
python3 -m venv venv
source venv/bin/activate          # Windows PowerShell: venv\Scripts\activate

# PyTorch: pick the build matching your setup (see `nvidia-smi` for CUDA version)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU only
python -m pip install -r requirements.txt

python -m src.pipeline --data-dir CMAPSSData --subset FD001          # ~1 min on GPU
python -m src.ablation --data-dir CMAPSSData --subset FD001          # 5 seeds, ~5-10 min on GPU
```

Training uses the GPU automatically when available (`--device auto|cpu|cuda`).
Latency is always measured on CPU at batch size 1, so it is comparable with
the CPU-only baselines.

Outputs land in `results/`:
- `results/FD001_summary.json`: all pipeline metrics (baselines, both
  surrogates, every detector and combiner, CIs, tuned thresholds)
- `results/FD001_ablation.md` / `.json`: the three ablation tables
- `results/figures/FD001_{mlp,bilstm}_training_curves.png`
- `results/figures/FD001_{mlp,bilstm}_rul_predictions.png`
- `results/figures/FD001_accuracy_vs_cost.png`

Or open `notebooks/01_pipeline_walkthrough.ipynb` to run interactively and
view the figures inline. Other subsets (`FD002`, `FD003`, `FD004`, which add
multiple operating conditions and/or a second fault mode) work with the same
`--subset` flag.

## Results

FD001, 17 features, window 30, RUL cap 125, fault = RUL ≤ 20, 40 epochs,
models trained on GPU. Test RMSE is on the last window of each of the 100
test engines, against RUL clipped at 125. Multi-seed numbers are mean ± std
over 5 seeds (the seed changes the train/val engine split, weight init and
batch order). Full outputs: `results/FD001_ablation.md`,
`results/FD001_summary.json`.

### 1. Surrogates vs. baselines: accuracy and cost

| Approach | Test RMSE | NMSE | PHM08 | Fault F1 (alarm: pred ≤ 20) | Train time | Latency (batch 1, CPU) | Params |
|---|---|---|---|---|---|---|---|
| Mean predictor | 40.48 | 1.020 | 17604 | 0.000 | none | ~0 | 1 |
| HI trend extrapolation | 37.19 | 0.861 | 18837 | 0.400 | fit only | 297 µs | 16 |
| HI similarity | 15.02 | 0.140 | 428 | 0.828 | fit only | 2633 µs | 16 |
| **MLP** | **14.25 ± 0.42** | 0.127 ± 0.008 | 328 ± 34 | 0.802 ± 0.028 | 6.1 ± 0.3 s | **75 ± 41 µs** | 73,729 |
| **BiLSTM** | **14.34 ± 0.59** | 0.128 ± 0.011 | 375 ± 68 | 0.823 ± 0.038 | 4.3 ± 0.8 s | 370 ± 192 µs | 46,657 |

- **The deep surrogates' advantage is mostly cost, not accuracy.** The
  similarity baseline, which needs no neural network, is within about
  0.7–0.8 RMSE of both models (bootstrap 95% CI for its RMSE: 12.5–17.4).
  But it scans all 100 training trajectories for every prediction, so the
  MLP answers **about 35× faster**, at a constant cost that does not grow
  with the size of the reference library. This is the accuracy/efficiency
  trade-off the project set out to measure.
- **MLP and BiLSTM are tied** within seed variance. A single seed-0 run
  suggested the BiLSTM was better (13.60 vs. 14.27), which multiple seeds
  showed to be noise. The BiLSTM is smaller but about 5× slower at batch
  size 1, because it processes timesteps sequentially.
- HI trend extrapolation performs poorly: degradation in C-MAPSS is flat
  early and accelerates late, so a local straight line mostly sees "no
  trend" and predicts the cap.

### 2. Preprocessing ablation (MLP)

| Variant | Test RMSE (clipped labels) | Test RMSE (true RUL) | PHM08 |
|---|---|---|---|
| Window 1 | 17.69 ± 0.20 | 18.74 ± 0.17 | 1059 ± 109 |
| Window 10 | 18.27 ± 0.30 | 19.18 ± 0.24 | 1126 ± 175 |
| **Window 30 (default)** | **14.25 ± 0.42** | **15.36 ± 0.52** | **328 ± 34** |
| Window 50 | 16.63 ± 0.60 | 17.47 ± 0.71 | 1239 ± 1190 |
| Window 30, no RUL clipping | 24.96 ± 0.99 | 24.73 ± 1.01 | 26866 ± 18869 |

- **RUL clipping is clearly justified**: without it RMSE rises from 14.3 to
  25.0, and it stays worse even when scored against the true, unclipped RUL.
- **Window 30 is best among those tested, but the trend is not clean.**
  Window 1 beats window 10, and window 50 is worse and unstable: the
  shortest test engines have only 31 cycles, so a 50-cycle window is mostly
  padding, and a few badly overestimated engines blow up the exponential
  PHM08 score. A likely confound is the fixed 40-epoch budget: the default
  MLP's validation loss was still falling at epoch 40, and larger inputs
  converge more slowly.

### 3. Leakage ablation (MLP)

| Split | Val RMSE | Test RMSE |
|---|---|---|
| Engine-level (default) | 14.65 ± 0.64 | 14.25 ± 0.42 |
| Window-level (leaky) | 15.13 ± 0.15 | 14.20 ± 0.48 |

The leaky split did **not** make validation look optimistic here. Leakage
only helps a model that can memorize near-duplicate windows; this MLP is
regularized with dropout and still underfitting (training MSE stays above
validation MSE), so it gains nothing from them. Engine-level splitting is
kept as the default because a higher-capacity model would exploit the leak.

### 4. Fault detection (seed 0)

Thresholds tuned on validation engines. "Last window" = 100 test engines, 16
faulty; "all windows" = every test window, 10,196 windows, 123 faulty.

| Detector (with MLP surrogate) | F1, last window [95% CI] | F1, all windows |
|---|---|---|
| Surrogate threshold (tuned: pred ≤ 19.4) | 0.812 [0.64, 0.94] | 0.727 |
| Isolation Forest | 0.733 [0.52, 0.88] | 0.640 |
| Rule-based (sensors > 3σ) | 0.839 [0.67, 0.96] | 0.651 |
| Surrogate OR rules | 0.857 [0.70, 0.96] | 0.665 |
| Surrogate AND rules | 0.786 [0.60, 0.93] | 0.718 |
| 2-of-3 vote | 0.759 [0.56, 0.91] | 0.781 |
| **Stacked logistic regression** | **0.867 [0.71, 0.97]** | **0.803** |

- **Learned score fusion (stacking) is the best detector** on the larger
  all-windows evaluation (0.803), ahead of voting (0.781) and every single
  detector.
- **On the last window alone, detectors cannot be ranked.** With only 16
  faulty engines, every confidence interval spans roughly ±0.15 and they
  all overlap.
- The validation-tuned surrogate alarm threshold (predicted RUL ≤ 19.4)
  landed close to the hand-picked 20, a useful sanity check.

## Limitations

- **Single dataset, simulated data.** FD001 has one operating condition and
  one fault mode; results on real engines would need domain adaptation and
  per-operating-condition normalization (relevant for FD002/FD004).
- **No physics model.** C-MAPSS ships none, so the baselines are classic
  data-driven prognostics methods, not physics-based estimators. The
  "surrogate" is trained on RUL labels, not on the outputs of an expensive
  simulator.
- **Fixed epoch budget.** The MLP had not converged at 40 epochs, which may
  confound the window-size comparison. Re-running with a larger budget
  (`--epochs 150`) and letting early stopping decide is the next step.
- **Detection results are single-seed**, and the last-window evaluation has
  only 16 positives; the all-windows numbers are the more reliable ones.
- **The stacked combiner is fit and threshold-tuned on the same validation
  windows**, which is mildly optimistic (the test set is still untouched).
- **Latency** is measured with Python-level timing on CPU; absolute values
  depend on hardware and vary run to run (see the ± values), but the
  relative ordering is stable.

### Changes from an earlier version

An earlier version compared against a single-sensor linear-extrapolation
baseline that used each test engine's last observed value as its "failure
level". That is invalid for truncated test engines, so it made the surrogate
look about 4.6× more accurate than it is against a fair baseline. It has
been replaced by the health-index baselines above. Detector thresholds,
previously set by hand, are now tuned on validation engines, and the
detectors use per-sensor features with a clean early-life reference; this
also reversed an earlier finding that OR-combining detectors hurts. That
finding came from a noisy rule detector, not from OR-combination in general.

## Notes / scope

- MATLAB reproduction of the final evaluation/plotting step is a nice-to-have
  extension, not implemented here. The metrics and figures are fully
  reproducible from `results/*.json` if needed.
- No web app or deployment layer, by design. This is a modeling/analysis
  portfolio piece.
- `src/windows.py` and `src/surrogate.py` are written to be dataset-agnostic
  (generic `group_col`/`time_col`/`feature_cols` arguments) so the
  sliding-window + surrogate approach can generalize to other sensor-stream
  problems (e.g. the hydraulic-systems condition-monitoring dataset
  mentioned as a fallback).
