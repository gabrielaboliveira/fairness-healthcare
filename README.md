# Fairness Treatment — In-Processing Bias Mitigation in Machine Learning

Experimental code for a master's dissertation (PPGC) on **mitigating bias in machine learning**, with a focus on **in-processing** methods. The pipeline trains baseline classifiers and bias-mitigation methods across several health-related datasets, then evaluates them on performance and a battery of group/individual fairness metrics. The analysis is organized around six research questions (RQ0–RQ5), from bias diagnosis to the fairness–performance trade-off.

## Overview

The study compares mitigation methods against unmitigated baselines under a unified evaluation protocol:

- **Baselines (4):** Logistic Regression and Random Forest, each in *aware* (uses the sensitive attribute) and *unaware* variants — `BLRA`, `BLRU`, `BRFA`, `BRFU`.
- **Pre-processing:** `EDL` — Disparate Impact Remover.
- **In-processing (main focus):**
  - `IGLA` / `IGLU` — Grid Search Reduction (*aware* / *unaware*)
  - `IP` — Prejudice Remover
  - `IFG` — FairGBM
  - `IAD` — Adversarial Debiasing
  - `IW` — Wasserstein Fair Classification

Results are aggregated with the median over folds, repetitions and the four base models to form the **baseline** reference. Mitigation effect is measured as the reduction in the distance to the metric's ideal value; statistical comparison uses a Friedman omnibus test followed by Holm post-hoc tests against the baseline.

## Datasets

Six UCI / public health datasets (downloaded via `ucimlrepo`), each with a defined sensitive attribute and privileged/unprivileged groups:

| Dataset | Sensitive attribute |
|---|---|
| Arrhythmia | sex |
| Mental | gender |
| Diabetes | gender |
| Heart Disease | sex |
| AIDS | homosexuality indicator (`homo`) |
| Obesity | gender |

Obesity sits near the performance ceiling and is frequently ineligible for mitigation under the gating criterion.

## Fairness metrics

Metrics are grouped into three categories:

- **A — Statistical parity:** Disparate Impact, Bias Amplification (DFBA)
- **B — Individual fairness:** Consistency, Generalized Entropy Index
- **C — Confusion matrix:** False Negative Rate Difference, False Discovery Rate Difference, Error Rate Difference

Each metric has a defined ideal value (e.g. Disparate Impact = 1, difference-based metrics = 0); the distance to that ideal drives the mitigation-effect computation. Performance is tracked via accuracy, F1, precision and recall, and data complexity via separability/overlap measures.

## Repository structure

```
fairnes_treatment/
├── auxiliar/                  # Experiment pipeline & shared helpers
│   ├── config.py              # Datasets, methods, metrics, ideal values, seeds
│   ├── experiments_pipeline.py# Training/evaluation routines (baseline, in-proc, pre-proc, adversarial)
│   └── utils.py               # Shared utilities (incl. complexity metrics)
├── auxiliar_analysis/         # Analysis & visualization helpers per research question
│   ├── rq0_utils.py … rq5_utils.py
│   ├── statistic_test.py      # Friedman + Holm post-hoc
│   ├── analysis_utils.py / dataset_*_*.py / structure.py
├── experiments/               # Notebooks and outputs
│   ├── experiments.ipynb      # Runs the experiments
│   ├── pre_processing.ipynb   # Pre-processing experiments
│   ├── analysis.ipynb         # Main RQ0–RQ5 analysis
│   ├── analysis_datasets.ipynb
│   └── results/               # Per-dataset results (gitignored)
├── data/                      # raw/ and processed/ data (gitignored)
├── aif360_local/              # Local AIF360 sources
├── fairgbm/                   # FairGBM (built locally)
├── wasserstein_fairness/      # Wasserstein Fair Classification
├── external/
├── requirements.txt
└── setup_fairgbm.bat          # Local FairGBM build
```

## Setup

The project targets **Python 3.12** (conda environment `fairness-research`).

```bash
# create/activate your environment, then:
pip install -r requirements.txt
```

Key dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `aif360`, `tensorflow` (for adversarial debiasing), `pot` (optimal transport, for Wasserstein), `ucimlrepo`, `pycol-complexity`, `tqdm`.

`fairgbm` and `wasserstein_fairness` are vendored in this repository (not installed via pip). On Windows, build FairGBM with:

```bash
setup_fairgbm.bat
```

> Optional: `graphviz` (Python package + system binary) is only needed for the RQ-structure diagram in `auxiliar_analysis/structure.py`.

## Running

1. **Run experiments** — open `experiments/experiments.ipynb` (and `pre_processing.ipynb` for the pre-processing method). Datasets are downloaded automatically; results are written to `experiments/results/<dataset>/`.
2. **Analyze results** — open `experiments/analysis.ipynb`, which produces the RQ0–RQ5 tables and figures used in the dissertation.

Datasets, sensitive attributes, active methods, cross-validation splits (5-fold), the train/test split (0.30) and the random seed (42) are all configured in `auxiliar/config.py`.

## Research questions

- **RQ0** — Bias diagnosis (which datasets/metrics exhibit bias)
- **RQ1** — Global mitigation effect
- **RQ2** — Mitigation by metric category (A/B/C)
- **RQ3** — Effect by bias severity
- **RQ4** — Effect by bias direction (against the unprivileged vs. the privileged group)
- **RQ5** — Fairness–performance trade-off

## Notes

Large data and result artifacts are gitignored (`data/raw/`, `data/processed/`, `experiments/results/`). This repository contains the experimental implementation only; the dissertation text, tables and figures live in a separate project.
