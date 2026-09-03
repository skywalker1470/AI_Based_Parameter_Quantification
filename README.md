
# AI-Based Fault Detection & Parameter Quantification on Engine Sensor Data

Portfolio project demonstrating surrogate modeling + fault/anomaly detection on
multivariate sensor time-series from a mechanical system, built as preparation
for AI-based fault detection roles on internal combustion engines.

## Motivation

Physics-based / exact estimation of an engine's health state (remaining
useful life, degradation level) from raw sensor streams is often expensive at
inference time (per-unit model fitting, iterative solvers, high-fidelity
simulation). A learned **surrogate model** can approximate that expensive
computation with a single amortized forward pass, at the cost of some
accuracy. This project quantifies that accuracy/cost trade-off and layers a
fault-detection mechanism on top of the surrogate's output.

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
  anomaly.py      - Isolation Forest + rule-based (threshold/deviation-count) fault detectors
  baseline.py     - "exact/physics-style" baseline: per-row expanding-window linear
                     extrapolation of a health sensor to its failure level
  evaluate.py     - RMSE, NMSE, PHM08 asymmetric score, precision/recall/F1
  plots.py        - training curves, predicted-vs-true RUL, accuracy-vs-cost plot
  pipeline.py     - end-to-end script wiring the above together
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

2. **Surrogate model**: a lightweight feed-forward network (flattened
   window fed into an MLP that outputs a RUL estimate) is the default; a
   small BiLSTM is also provided (`src/surrogate.py`) as a sequence-aware
   alternative, and is demoed in the notebook. The surrogate stands in for
   an expensive physics-based computation.

3. **Fault/anomaly detection layer**, built on top of the surrogate:
   - **Isolation Forest** on raw sensor windows, fit on windows known to be
     healthy (RUL above the fault threshold).
   - **Rule-based detector**: flags a window as faulty when enough sensors
     simultaneously deviate beyond N standard deviations from a healthy
     baseline, a simple, interpretable analogue to an association-rule
     (FP-Growth-style) "co-occurring deviation" approach.
   - **Surrogate-threshold**: flag fault when the surrogate's predicted RUL
     drops below the threshold.
   - **Hybrid**: union of surrogate-threshold and rule-based flags.

4. **Ablation study** compares three approaches on accuracy (RMSE/NMSE of RUL,
   F1 of fault detection) and cost (training time, per-sample inference time):
   - *Exact/physics-style baseline*: per-row expanding-window linear
     regression of a health-indicator sensor, extrapolated to its
     failure level. Accurate in principle, but must be recomputed per
     row/engine, so it is far more expensive per prediction than the
     amortized surrogate.
   - *Surrogate-only (DNN)*.
   - *Surrogate + rule/statistical layer* (hybrid detection).

5. **Evaluation**: RMSE and NMSE for RUL (parameter quantification),
   precision/recall/F1 for fault detection, plus the PHM08 competition's
   asymmetric scoring function (penalizes late predictions more than early
   ones), and an accuracy-vs-cost scatter plot per approach.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

python -m src.pipeline --data-dir CMAPSSData --subset FD001 --epochs 40
```

Outputs land in `results/`:
- `results/FD001_summary.json`: all metrics (surrogate accuracy, detection
  precision/recall/F1 per method, exact-baseline comparison)
- `results/figures/FD001_training_curves.png`
- `results/figures/FD001_rul_predictions.png`
- `results/figures/FD001_accuracy_vs_cost.png`

Or open `notebooks/01_pipeline_walkthrough.ipynb` to run interactively and
view the figures inline. Other subsets (`FD002`, `FD003`, `FD004`, which add
multiple operating conditions and/or a second fault mode) work with the same
`--subset` flag.

## Results

FD001, 40 epochs, window size 30, RUL cap 125, fault threshold RUL <= 20.
Full numbers in `results/FD001_summary.json`.

| Approach | RMSE (RUL) | NMSE | Detection F1 | Train time | Inference / sample |
|---|---|---|---|---|---|
| Exact/physics-style baseline | 70.89 | 3.129 | n/a | 0s (no training) | 14.3 us |
| Surrogate-only (DNN) | 15.40 | 0.148 | 0.813 (surrogate-threshold) | 7.5s | 3.8 us |
| Surrogate + rule layer (hybrid) | 15.40 | 0.148 | 0.667 | 7.5s | 3.8 us |

Detection F1 by method (all evaluated on the same test set, fault = RUL <= 20):

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| Isolation Forest | 0.593 | 1.000 | 0.744 |
| Rule-based (deviation count) | 0.556 | 0.938 | 0.698 |
| Surrogate-threshold | 0.813 | 0.813 | 0.813 |
| Hybrid (surrogate OR rules) | 0.517 | 0.938 | 0.667 |

**What the ablation shows:**

- The exact/physics-style baseline is both far less accurate (RMSE 70.9 vs.
  15.4) and slower per prediction (14.3 us vs. 3.8 us) than the DNN
  surrogate. It needs a fresh linear regression fit per row, so cost scales
  with trajectory length, while the surrogate is trained once and then
  infers in constant time. This is the core accuracy/efficiency trade-off
  the project set out to quantify: the surrogate wins on both axes here,
  because the naive physics-style baseline is a weak approximation of true
  engine physics, not because surrogates are free of that trade-off in
  general.
- The surrogate-threshold detector (flag fault when predicted RUL drops
  below 20) is the strongest detector on its own, with balanced precision
  and recall (0.81 / 0.81). Isolation Forest and the rule-based detector
  both favor recall over precision (they catch nearly every fault but with
  more false alarms), which is typical for unsupervised/threshold methods
  tuned without access to labels.
- The hybrid detector (union of surrogate-threshold and rule-based flags)
  underperforms both of its inputs (F1 0.667 vs. 0.813 and 0.698). Taking
  the OR of two detectors compounds their false positives without adding
  recall the surrogate didn't already have, so precision drops sharply
  (0.517). This is a genuine, useful finding from the ablation rather than
  a bug: naive OR-combination is not a free win, and a better hybrid would
  need AND-logic, a voting scheme, or a learned combiner instead of a
  simple union.

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
