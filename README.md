# Risk-Calibrated Belief-State MDP for Predictive Maintenance

> **From Prognosis to Action: Risk-Calibrated Belief-State MDPs with Temporal Attention for Predictive Maintenance**
> Whashak Faeid, Omer Tahsin — Bangladesh University of Engineering & Technology

---

## Overview

Predictive maintenance (PdM) systems routinely estimate Remaining Useful Life (RUL) from sensor data but then discard the uncertainty in those estimates before making decisions. This project shows that treating predicted health labels as ground truth inflates the Critical→Near-Failure transition probability by **6.1× to 13.4×** across the NASA CMAPSS benchmark — systematically miscalibrating policies near the failure boundary, exactly where the cost of error is highest.

This work reframes PdM as a **belief-state Markov Decision Process (MDP)**: instead of acting on a single predicted label, the agent maintains a probability distribution over health states and selects actions to maximise expected cumulative reward under that distribution.

---

## Key Contributions

- **Transition distortion analysis** — Quantifies, for the first time on the full CMAPSS benchmark, how hard-label MDPs become miscalibrated near the failure boundary.
- **TABiE (Temporal Attention Belief Encoder)** — A lightweight two-layer Transformer that maps a 30-cycle sensor window directly to a calibrated health-state belief vector, achieving Near-Failure recall above **85%** on all four sub-datasets (97.5% on FD001).
- **Belief temperature calibration** — A scalar temperature parameter tuned exclusively on training engines sharpens over-dispersed belief vectors, balancing safety and economic performance without accessing holdout data.
- **Risk-calibrated evaluation** — Separates economic net reward from unsafe Near-Failure exposure, producing a parametric safety–economics trade-off curve and identifying the penalty crossover C\*ᵤ ≤ 5.0 on three of four sub-datasets.
- **Statistical validation** — Paired Wilcoxon signed-rank tests confirm the Calibrated Belief MDP reduces unsafe cycles significantly (p = 0.015, p = 0.027) with no significant economic cost (p > 0.36) under both Ridge and Random Forest predictors.

---

## Method Summary

```
NASA CMAPSS → Preprocessing → RUL Prediction (Ridge / RF)
                                        ↓
                          Belief Construction
                         ┌────────────────────┐
                         │  Gaussian Baseline  │  CDF over per-state σ
                         │  TABiE (proposed)   │  2-layer Transformer
                         └────────────────────┘
                                        ↓
                    Belief-State MDP  →  Value Iteration
                                        ↓
                      Event-Based Evaluation on Holdout Engines
                      (Risk-adjusted reward Jπ  +  unsafe NF cycles)
```

**Health states:** Healthy (RUL > 100) · Degrading (50–100) · Critical (15–50) · Near-Failure (≤ 15)  
**Actions:** Do-Nothing · Preventive Maintenance · Replacement  
**Datasets:** CMAPSS FD001 – FD004 (100–260 engines, 1–6 operating conditions)

---

## Results at a Glance

| Dataset | Oracle MDP n̄_NF | Practical MDP n̄_NF | Calibrated Belief MDP n̄_NF |
|---------|:-----------:|:---------------:|:----------------------:|
| FD001   | 0.00        | 1.55            | **0.70**               |
| FD002   | 0.00        | 1.23            | **1.04**               |
| FD003   | 0.00        | 1.85            | **0.90**               |
| FD004   | 0.00        | 2.62            | **1.80**               |

*Results under Random Forest predictor. Lower n̄_NF = fewer unsafe operating cycles.*

---

## Repository Structure

```
├── notebook/
│   └── belief_state_mdp.ipynb       # Full pipeline: preprocessing → TABiE → MDP → evaluation
├── data/                            # Place CMAPSS .txt files here (see below)
├── figures/
│   └── tradeoff_curve.png           # Risk-adjusted reward vs unsafe-cycle penalty
└── README.md
```

---

## Data

This project uses the publicly available **NASA CMAPSS Turbofan Engine Degradation Simulation** dataset.

Download the four sub-dataset files (`train_FD001.txt` – `train_FD004.txt`, `test_FD001.txt` – `test_FD004.txt`, `RUL_FD001.txt` – `RUL_FD004.txt`) from either source below and place them in the `data/` folder:

- **NASA PCoE Repository:** https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- **Kaggle mirror:** https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

---

## Requirements

```bash
pip install numpy pandas scikit-learn torch matplotlib scipy
```

Python 3.9+ · PyTorch 2.0+

---

## Citation


