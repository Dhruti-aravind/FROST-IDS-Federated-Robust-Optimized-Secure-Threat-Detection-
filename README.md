# FROST-IDS: Federated Robust Optimized Secure Threat Detection

**Federated IDS framework with Mixture-of-Experts, FPR-Constrained Threshold Calibration, and performance-weighted aggregation — evaluated on UNSW-NB15.**

---

## Overview

FROST-IDS is a federated intrusion detection system architected around deployment-oriented false-positive rate (FPR) enforcement rather than unconstrained aggregate performance. It integrates three coordinated components:

1. **Mixture-of-Experts (MoE) Classifier** — Three structurally distinct expert sub-networks with specialized focal-loss objectives, governed by Gumbel-Softmax temperature annealing.
2. **Performance-and-Size-Weighted FedAvg Aggregator** — FPR compliance bonuses steer gradient updates toward clients that satisfy the FPR budget.
3. **Binary-Search Threshold Calibration** — Conservative calibration target (0.48%), 25% distribution-shift safety margin, and a 99.5th-percentile quantile guard.

On UNSW-NB15 across seven random seeds:

| Metric | Value |
|---|---|
| Accuracy | 84.6% ± 0.8% |
| Recall | 72.9% ± 1.5% |
| FPR | 0.97% ± 0.06% |
| AUC-ROC | 0.960 ± 0.001 |
| FPR ≤ 1% compliance | 5/7 seeds (vs. 3/7 for FedAvg) |
| FPR compliance under severe non-IID (α=0.1) | 4/4 seeds (vs. 3/4 for FedAvg) |

---

## Motivation

A medium-sized enterprise network processing 500,000 flows per hour at a 1% FPR produces over 5,000 false positives per hour — more than one per second — forcing SOC analysts to spend the vast majority of their time investigating non-threats. Despite this, almost all existing IDS evaluations optimize overall accuracy or F1-score on balanced test splits without imposing an explicit FPR budget.

FROST-IDS treats IDS deployment as a **risk-budget allocation problem**: the primary success criterion is per-seed FPR compliance, with recall and accuracy as secondary constraints.

---

## Architecture

### Shared Encoder
- Input: 34 UNSW-NB15 flow features
- Architecture: `Dense(256, GELU) → BN → Dropout(0.25) → Dense(128, GELU) → BN`
- L2 regularization (λ = 3×10⁻⁴) on all dense kernels
- Output: 128-dimensional embedding

### Expert Sub-Networks

| Expert | Activation | Focal Parameters | Inductive Bias |
|---|---|---|---|
| Expert R (Recall) | ReLU | γ=3.0, α=0.97 | Maximizes sensitivity to rare attack events |
| Expert P (Precision) | tanh | γ=0.5, α=0.35 | Keeps sub-network in the low-FPR region |
| Expert A (Anomaly) | ReLU + residual skip | γ=2.0, α=0.80 | Preserves raw distributional data for structural novelties |

### Difference-Augmented Gate
- Input: `[encoder_embedding (128); logit_R; logit_P; logit_A; |logit_R − logit_P|; |logit_R − logit_A|; |logit_P − logit_A|]` → 135-dimensional vector
- Gumbel-Softmax with top-2 sparsification
- Temperature anneals from τ=8.0 → τ=0.3 (factor 0.78 per checkpoint)
- Gate spread: Δg ≈ 0.12 at checkpoint 0 → 0.43 at checkpoint 8

### Training Objective

```
L = L_main + 0.35·L_R + 0.20·L_P + 0.25·L_A
    + λ_H·H(g) + λ_div·L_div + λ_bal·L_bal
```

- `L_main`: focal cross-entropy (γ=2.0, ε_smooth=0.01) with inverse square-root class frequency weights
- `H(g)`: gate entropy regularizer (λ_H = 5×10⁻³)
- `L_div` (weight 0.05): expert diversity
- `L_bal` (weight 0.01): load balance

---

## Data Partitioning

### Two-Phase Dirichlet Allocator (α=0.3)
- **Phase 1**: Algebraic 15% class-floor guarantee per client (no rejection sampling). Ensures no client has zero attack samples.
- **Phase 2**: Remaining mass drawn from Dirichlet(α) for realistic heterogeneity.
- Fallback to rejection sampling at α=0.1 when the algebraic floor fails.

### SMOTE Augmentation
- Per-class ratio cap: 12×, triggered below 35% class frequency
- Gaussian noise corruption: σ=0.08 (feature-wise scaled), raised from 0.03 to reduce cluster redundancy in Fuzzers, Shellcode, and Worms
- Loss weights: inverse square-root class frequency, range [0.33, 5.73]

---

## Federated Aggregation

Client weights are computed as:

```
w_k = 0.65 · φ_k + 0.35 · (n_k / n_total)
```

Where `φ_k` is the FPR-penalized validation accuracy:

```
φ_k ← φ_k · max(0.01, 1 − 10 · max(0, FPR_k − 0.01))
```

Clients meeting FPR ≤ 1% receive a **1.3× compliance bonus**. Weights are computed independently for the encoder, each expert, and the gating network.

---

## Threshold Calibration

### Algorithm: Binary-Search FPR-Constrained Calibration

```
Input:  Model f_θ, calibration set D_cal, target τ_cal = 0.0048, quantile q = 0.995
Output: Decision threshold θ*

1. Compute predicted probabilities p̂_i = f_θ(x_i) for all (x_i, y_i) ∈ D_cal
2. Find T* = argmin_{T ∈ [0.3, 3.0]} ECE(p̂) + 0.005·(T − 1)²   [L2-regularized temperature scaling]
3. Re-scale: p̂_i ← σ(log(p̂_i / (1 − p̂_i)) / T*)
4. Guard: θ_q ← Q_{0.995}({p̂_i | y_i = 0})
5. Binary search on [0, 1] (tolerance 10⁻⁴) for smallest θ with FPR(θ) ≤ τ_cal
6. θ* ← max(θ_binary, θ_q)
```

The conservative 0.48% calibration target + 25% safety margin covers the observed +0.003–+0.006 calibration-to-test FPR shift under non-IID splits.

---

## Experimental Setup

| Setting | Value |
|---|---|
| Dataset | UNSW-NB15 (257,673 samples, 34 features, 9 attack categories) |
| Split | 70% train / 10% calibration / 20% test (stratified) |
| Calibration attack prevalence | ~55% |
| Federated clients | 5 |
| Federated rounds | 18 |
| Evaluation checkpoints | {1, 3, 5, 7, 9, 11, 13, 15, 17} |
| Optimizer | AdamW (η=0.001, wd=3×10⁻⁴, cosine LR schedule to η_min=0.0001) |
| Batch size | 512 |
| Seeds | 42, 123, 456, 789, 1011, 2024, 3141 |
| Hardware | Tesla P100-PCIE-16GB |
| Framework | TensorFlow 2.19.0 |

---

## Results

### Main Comparison (7 seeds, mean ± 95% CI)

| Method | Acc (%) | Rec (%) | FPR (%) | FPR-S | AUC | ECE | ExpCost |
|---|---|---|---|---|---|---|---|
| Simple MLP | 84.0±0.1 | 71.5±0.2 | 0.72±0.1 | 7/7 | 0.960 | 0.103 | 64,759 |
| FedAvg | 84.4±0.6 | 72.5±1.2 | 0.98±0.1 | 3/7 | 0.959 | 0.116 | 62,710 |
| No MoE | 84.5±0.6 | 72.6±1.1 | 0.86±0.1 | 7/7 | 0.961 | 0.127 | 62,530 |
| No Size Weight | 84.5±0.6 | 72.7±1.0 | 0.99±0.1 | 3/7 | 0.959 | 0.114 | 62,158 |
| Centralized (upper bound) | 89.4±0.2 | 81.6±0.4 | 0.99±0.1 | 4/7 | 0.972 | 0.114 | 42,174 |
| **FROST-IDS** | **84.6±0.8** | **72.9±1.5** | **0.97±0.06** | **5/7** | **0.960** | **0.113** | **61,779** |

> FPR-S = seeds with FPR ≤ 1%. Centralized is a privacy-violating upper bound, shown for reference only. FROST-IDS achieves the lowest ExpCost among all federated variants.

### Heterogeneity Stress Test (α=0.1 vs α=0.3, 4 seeds)

| Method | Acc α=0.3 | Rec α=0.3 | FPR-S α=0.3 | Acc α=0.1 | Rec α=0.1 | FPR-S α=0.1 |
|---|---|---|---|---|---|---|
| FedAvg | 84.4% | 72.5% | 3/4 | 86.5% | 76.2% | 3/4 |
| **FROST-IDS** | **84.6%** | **72.9%** | **4/4** | **86.9%** | **76.9%** | **4/4** |

### Per-Attack-Category Recall (seed-0)

| Category | Test n | Recall | ≥65%? |
|---|---|---|---|
| Generic | 18,871 | 0.98 | Pass |
| Backdoor | 583 | 0.93 | Pass |
| DoS | 4,089 | 0.83 | Pass |
| Exploits | 11,132 | 0.65 | Near floor |
| Reconnaissance | 3,496 | 0.61 | Near floor |
| Fuzzers | 6,062 | 0.24 | Below floor |
| Shellcode | 378 | 0.39 | Below floor |
| Worms | 44 | 0.34 | Below floor |

---

## Ablation Study Findings

- Removing **performance-weighted aggregation** is the single largest driver: FPR-S drops from 5/7 to 3/7.
- Removing **size weighting** while keeping performance weighting also yields 3/7, confirming both terms in Eq. (3) are necessary.
- Removing **MoE** achieves 7/7 compliance mechanically (lower score variance) but increases ECE from 0.113 to 0.127 and raises expected deployment cost by 751 units.
- Full FROST-IDS achieves the **lowest expected deployment cost** among all federated variants, confirming decision-theoretic superiority.

---

## XAI Analysis

Four attribution methods were evaluated without exposing raw client traffic:

| Method pair | Jaccard (top-10) | Spearman ρ |
|---|---|---|
| Permutation – Integrated Gradients | 0.818 | 0.83 |
| SHAP – Permutation | 0.538 | — |
| IG – SHAP | 0.667 | — |

**Consistently top-ranked features across all four methods**: `dwin`, `state`, `swin`, `ackdat`, `dload`.

Domain interpretation:
- `dwin`, `swin`: destination/source TCP window sizes — indicative of volumetric attacks and flow-control anomalies
- `state`, `ackdat`: connection establishment patterns
- `service`: protocol-level behavior
- `synack`: sole normal-biased feature (legitimate TCP three-way handshake)

---

## Robustness

- **Gaussian noise**: Accuracy, F1, and Recall stable through σ ≤ 0.15; FPR compliance maintained through σ = 0.15 and breached above σ = 0.20. This defines the operational noise tolerance boundary.
- **Convergence**: Accuracy and F1-Macro converge by checkpoint 5 (round 9). FPR compliance in 5/7 seeds from checkpoint 1 onwards. Seed band width < 0.02 across all panels.

---

## Calibration Details

- Temperature scaling: ECE improves from 0.136 → 0.113 (mean T* = 0.789, λ = 0.005)
- Calibrated threshold (seed-0): θ* = 0.864
- ECE (seed-0): 0.112
- Threshold anchored in the high-confidence region (mean predicted probability > 0.85 for classified positives), making FPR less sensitive to small score perturbations at deployment

---

## Limitations

- **FPR compliance**: 2 of 7 seeds remain above the 1% target. Reducing τ_cal from 0.0048 to 0.0035 is expected to achieve 7/7 compliance at a moderate recall cost; the empirical calibration-to-test FPR shift is bounded by +0.006.
- **Minority-class recall**: Fuzzers, Shellcode, and Worms fall below the 65% recall floor. This reflects the fundamental tension between strict FPR ceilings and sensitivity to rare classes with overlapping feature signatures — not solely an implementation deficiency.
- **Dataset scope**: Primary experiments use UNSW-NB15 only. A CIC-IDS2017/2018 probe at 50% feature coverage showed high FPR due to feature mismatch; no cross-dataset generalization is claimed.

---

## Future Work

1. **Stricter calibration goal**: Reduce τ_cal to 0.0035 to target 7/7 FPR compliance with low recall overhead.
2. **Per-category oversampling**: Adaptive k-NN noise augmentation pipelines to elevate Fuzzers, Shellcode, and Worms recall above 65%.
3. **Differential privacy**: Rigorous (ε, δ)-DP treatment of gradient exchange with per-client gradient clipping.
4. **Communication efficiency**: Gradient compression via top-k sparsification or quantization-aware training for mobile edge scenarios.
5. **Online re-calibration**: Sliding-window production monitor with incremental binary search updates to θ* for non-stationary traffic.

---

## Dependencies

```
tensorflow==2.19.0
numpy==2.0.2
pandas==2.3.3
```

---

## Dataset

UNSW-NB15 is publicly available from the University of New South Wales Canberra Cyber Range Laboratory. After preprocessing, 34 features are retained spanning:

- **Flow features**: `dur`, `proto`
- **Content features**: `sload`, `dload`
- **Time features**: `tcprtt`, `synack`
- **General-purpose indicators**: `state`, `service`

The identifier column and redundant columns are removed prior to training. All standardization is fit exclusively on the training split.

---
