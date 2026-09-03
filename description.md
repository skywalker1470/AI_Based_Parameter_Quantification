# Project: AI-Based Fault Detection / Parameter Quantification (Engine Sensor Data)

## Goal
Build a small, well-documented pipeline that demonstrates surrogate modeling +
anomaly/fault detection on multi-sensor time-series data from a mechanical
system, as a portfolio piece for a Scania/Traton thesis application on
"AI-based fault detection on an internal combustion engine."

## Dataset (pick one, prefer the first if accessible)
1. NASA C-MAPSS Turbofan Engine Degradation Simulation dataset
   (multivariate sensor time-series, run-to-failure, labeled degradation).
2. UCI / Kaggle "Condition Monitoring of Hydraulic Systems" dataset
   (multi-sensor, labeled fault states — good fallback if C-MAPSS is awkward).

## Pipeline
1. **Data prep**: load sensor streams, sliding-window segmentation
   (window size / stride configurable), train/val/test split that respects
   time order (no leakage across a run).
2. **Surrogate model**: lightweight feed-forward DNN (or small BiLSTM if
   time allows) that learns to estimate an operating parameter or health
   indicator from raw sensor inputs, standing in for an expensive
   physics-based / exact computation.
3. **Fault/anomaly detection layer** on top of the surrogate output:
   - Baseline: Isolation Forest or a rule-based approach analogous to the
     FP-Growth association-rule pipeline used in an earlier building-heating
     anomaly-detection project.
   - Compare against the DNN surrogate-based approach.
4. **Ablation study**: compare "exact/physics-style baseline" vs.
   "surrogate-only" vs. "surrogate + rule/statistical layer" on accuracy
   and computational cost (inference time, training cost). This mirrors an
   ablation structure already used in a prior P-Center surrogate-modeling
   project (exact vs. surrogate vs. hybrid).
5. **Evaluation**: precision/recall/F1 for fault detection, RMSE or NMSE for
   parameter quantification, plus a plot of the accuracy-vs-cost trade-off.
6. **Optional**: reproduce the final evaluation/plotting step in MATLAB,
   since the target thesis role lists MATLAB experience as a plus.

## Deliverables
- Clean, documented Python project (notebook + scripts), pushed to GitHub.
- A short README: motivation, dataset, method, results, and what the
  ablation shows about the accuracy/efficiency trade-off.
- Keep scope tight — this is a 2-3 week side project, not a full thesis.

## Constraints / preferences
- Prefer PyTorch (primary DL framework already in use).
- Keep code modular so the sliding-window / surrogate-model parts could
  later generalize to other sensor-stream problems (reusable, not
  one-off).
- No need for a web app or deployment layer — this is a modeling/analysis
  portfolio piece, not a product.