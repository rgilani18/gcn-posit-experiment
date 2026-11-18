# GCN Posit16 vs FP16 — Numerical Precision Study

> **A bare-metal C++ experiment comparing `posit<16,1>` (Posit16) against IEEE 754 `cfloat<16,5>` (FP16) across three training regimes of a two-layer Graph Convolutional Network on the Cora citation dataset — with every result audited against an 80-bit long double ground truth.**

---

## Table of Contents

1. [Overview](#overview)
2. [Key Findings](#key-findings)
3. [Number Format Comparison](#number-format-comparison)
4. [The Three Training Regimes](#the-three-training-regimes)
5. [Four FP16 Failure Modes](#four-fp16-failure-modes)
6. [Numerical Accuracy Audit](#numerical-accuracy-audit)
7. [Dataset — Cora](#dataset--cora)
8. [Project Structure](#project-structure)
9. [Quick Start](#quick-start)
10. [Reports and Documentation](#reports-and-documentation)

---

## Overview

This experiment investigates whether **Posit16** — a modern alternative to IEEE 754 floating-point — can train a Graph Convolutional Network more reliably than **FP16** when both formats are constrained to 16 bits.

Three distinct training configurations (regimes) are tested:

| Regime | Optimizer | Moment Storage | Learning Rate |
|--------|-----------|----------------|---------------|
| **Regime 1** | Vanilla Gradient Descent | — | 0.5 |
| **Regime 2** | Pure 16-bit Adam | 16-bit (same format as weights) | 0.01 |
| **Regime 3** | Mixed-Precision Adam | 64-bit double | 0.01 |

All experiments share identical hyperparameters, the same random seed (`42`), the same Xavier-initialized weights, and the same Cora graph. The only variable is the number format.

After training, every model's final weights are evaluated against an **80-bit long double** reference — giving a precision-independent ground truth to measure true numerical accuracy.

---

## Key Findings

### Results Table — 200 Epochs on Cora

| Regime | Format | Test Accuracy | Val Loss | MAE vs 80-bit | RMSE vs 80-bit | Peak Error | Disagreements |
|--------|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| R1 — Vanilla GD | FP16 | 19.30% | 1.9343 | 5.237×10⁻⁴ | 8.675×10⁻⁴ | 0.0432 | 21 / 2708 |
| R1 — Vanilla GD | **Posit16** | **27.60%** | **1.9124** | **4.363×10⁻⁵** | **6.175×10⁻⁵** | **0.0019** | **2 / 2708** |
| R2 — Pure 16-bit Adam | FP16 | 43.60% | 6.9872 ⚠️ | 9.319×10⁻³ | 1.284×10⁻² | 0.2189 | 1 / 2708 |
| R2 — Pure 16-bit Adam | **Posit16** | **77.80%** | **0.8066** | **2.810×10⁻⁴** | **4.411×10⁻⁴** | **0.0101** | **1 / 2708** |
| R3 — Mixed Double Adam | FP16 | 71.20% | 1.2453 | 3.322×10⁻³ | 4.860×10⁻³ | 0.1660 | 6 / 2708 |
| R3 — Mixed Double Adam | **Posit16** | **80.20%** | **0.8320** | **2.252×10⁻⁴** | **3.485×10⁻⁴** | **0.0119** | **0 / 2708** ✅ |

> **⚠️ Regime 2 FP16 loss = 6.9872** — catastrophic underflow in Adam caused the second moment to collapse, producing exploding weight updates. Posit16 trained stably to 0.8066.

> **✅ Posit16 Regime 3 disagreements = 0** — 100% exact agreement with 80-bit ground truth on all 2,708 test nodes. FP16 still produced 6 silent misclassifications under the same conditions.

### Accuracy Gap Summary

```
Regime 1  │ FP16  19.30%  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
           │ Posit 27.60%  ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (+8.3 pp)

Regime 2  │ FP16  43.60%  █████████████████░░░░░░░░░░░░░░░░░░░░░░░
           │ Posit 77.80%  ███████████████████████████████░░░░░░░░░  (+34.2 pp)

Regime 3  │ FP16  71.20%  ████████████████████████████░░░░░░░░░░░░
           │ Posit 80.20%  ████████████████████████████████░░░░░░░░  (+9.0 pp)
```

---

## Number Format Comparison

### Bit Layout

```
FP16  — cfloat<16,5>:
  [S][E E E E E][M M M M M M M M M M]
   1      5             10             = 16 bits

Posit16 — posit<16,1>:
  [S][regime bits...][E][fraction bits...]
   1   variable      1    variable        = 16 bits
```

### Range and Precision

| Property | FP16 `cfloat<16,5>` | Posit16 `posit<16,1>` |
|----------|--------------------|-----------------------|
| Largest value | 65,504 | 268,435,456 |
| Smallest normal | 6.10 × 10⁻⁵ | — (no normal/subnormal split) |
| Smallest positive | **5.96 × 10⁻⁸** (subnormal) | **3.72 × 10⁻⁹** |
| Mantissa bits (near 1.0) | Fixed **10 bits** | Tapered up to **14 bits** |
| ULP near 1.0 | 9.77 × 10⁻⁴ | 6.10 × 10⁻⁵ |
| Precision ratio near 1.0 | 1× | **16× finer** |
| Subnormal zone | Yes — degrades near floor | **No — full precision to minimum** |
| Special values | ±∞, NaN, ±0 | Only 0 and NaR |

### Key Insight

Posit's **tapered encoding** automatically allocates more fraction bits where neural network values are densest — near ±1.0. FP16 uses a fixed-width mantissa that cannot adapt, leaving it coarse precisely where it matters most.

---

## The Three Training Regimes

### Regime 1 — Vanilla Gradient Descent

The simplest optimizer. Every weight is updated by a fixed step in the direction of the gradient:

```
w ← w − lr × ∇w          (lr = 0.5)
```

No momentum, no adaptive scaling. FP16's accumulated rounding across 13,264 edges causes the larger drift seen in Regime 1 (21 disagreements vs Posit's 2).

**Code:** [`src/cpp/train_2layer_gd_200ep.cpp`](experiment_200ep_3regimes/src/cpp/train_2layer_gd_200ep.cpp)

---

### Regime 2 — Pure 16-bit Adam

Adam tracks two moment estimates per weight — both stored entirely in 16-bit:

```
mₜ = β₁·mₜ₋₁ + (1−β₁)·gₜ          (β₁ = 0.9)
vₜ = β₂·vₜ₋₁ + (1−β₂)·gₜ²         (β₂ = 0.999)
w  ← w − lr · (m̂ₜ / (√v̂ₜ + ε))    (lr = 0.01, ε = 1×10⁻⁴)
```

**The critical failure point for FP16:**

Near convergence, gradients shrink (e.g. gₜ = 0.005). The new second-moment contribution becomes:

```
(1 − β₂) × gₜ² = 0.001 × (0.005)² = 2.5 × 10⁻⁸
```

| Format | Minimum representable | 2.5×10⁻⁸ | Outcome |
|--------|-----------------------|-----------|---------|
| FP16 | 5.96 × 10⁻⁸ | Below floor | **Flush to 0 → vₜ collapses → update explodes** |
| Posit16 | 3.72 × 10⁻⁹ | Within range | **Stored correctly → stable convergence** |

Result: **FP16 loss = 6.9872 (diverged), Posit16 loss = 0.8066 (converged).**

**Code:** [`src/cpp/train_2layer_pure16_adam_200ep.cpp`](experiment_200ep_3regimes/src/cpp/train_2layer_pure16_adam_200ep.cpp)

---

### Regime 3 — Mixed-Precision Double Adam

Adam moment vectors (`m`, `v`) are stored in **64-bit double** (floor ≈ 5×10⁻³²⁴), eliminating underflow. Weights remain in 16-bit.

```
// Moment accumulation: 64-bit double (no underflow)
v_d += (1 - beta2_d) * g_d * g_d;

// Weight update: cast back to 16-bit
w[i] = T(static_cast<double>(w[i]) - step);
```

This fixes the Regime 2 collapse (FP16 jumps from 43.60% → 71.20%), but FP16's coarse 10-bit mantissa still rounds away fine weight updates — explaining why Posit16 maintains a 9 pp lead (80.20%) with no 64-bit memory overhead at all.

**Code:** [`src/cpp/train_2layer_double_adam_200ep.cpp`](experiment_200ep_3regimes/src/cpp/train_2layer_double_adam_200ep.cpp)

---

## Four FP16 Failure Modes

### 1 — Catastrophic Underflow in Adam
Small squared gradients `(1−β₂)·gₜ²` flush to 0 in FP16, causing the second-moment denominator to collapse and weight updates to explode by up to **100×**. Posit16 represents these values natively.
**Impact: FP16 loss 8.67× worse in Regime 2 (6.9872 vs 0.8066).**

### 2 — Low Mantissa Precision Near 1.0
FP16's fixed 10-bit mantissa gives ULP ≈ 9.77×10⁻⁴ near 1.0. Typical late-training Adam steps (~3.3×10⁻⁴) are smaller than this ULP — they are rounded away. Posit16's tapered 14-bit fraction gives ULP ≈ 6.10×10⁻⁵, preserving every update.
**Impact: FP16 logit MAE is 33× higher than Posit16 in Regime 2.**

### 3 — Message-Passing Drift
Each of the 13,264 edge aggregation operations introduces rounding error. Across 212,224 total additions per forward pass (13,264 edges × 16 channels), errors compound across two graph layers.
**Impact: FP16 peak logit error = 0.0432 vs Posit16 = 0.0019 — 22.98× higher drift.**

### 4 — Silent Prediction Flips
When accumulated drift exceeds the margin between the top-2 predicted classes, the predicted label flips — silently, with no warning in the loss.
**Impact: 21 silent misclassifications in Regime 1 (FP16) vs 2 (Posit16). Zero for Posit16 in Regime 3.**

---

## Numerical Accuracy Audit

The audit engine runs every trained model's final weights through **two identical forward passes**:

1. **16-bit pass** — in the training format (FP16 or Posit16)
2. **80-bit pass** — same weights upcast to `long double` (18 decimal digits of precision)

The difference between the two is pure numerical error from the 16-bit format.

```
Metrics computed across 18,956 logit values (2,708 nodes × 7 classes):

  MAE        = mean |logit₁₆ − logit₈₀|
  RMSE       = √ mean (logit₁₆ − logit₈₀)²
  Max Error  = max |logit₁₆ − logit₈₀|
  Disagreements = nodes where argmax(logit₁₆) ≠ argmax(logit₈₀)
```

**Why 80-bit long double?**
Its accumulated rounding error over 13,264 edge additions is ~10⁻¹⁵ — over 10¹¹× smaller than the errors being measured in 16-bit formats. It is the most precise standard C++ type available without external libraries, used here as a mathematically rigorous reference.

**Audit code:** [`src/cpp/evaluate_3regimes_numerical_accuracy.cpp`](experiment_200ep_3regimes/src/cpp/evaluate_3regimes_numerical_accuracy.cpp)
**Results CSV:** [`results/logs/numerical_accuracy_3regimes.csv`](experiment_200ep_3regimes/results/logs/numerical_accuracy_3regimes.csv)

---

## Dataset — Cora

| Property | Value |
|----------|-------|
| Dataset | Cora Citation Network |
| Nodes (N) | 2,708 |
| Features per node (F) | 1,433 (bag-of-words) |
| Directed edges (E) | 13,264 |
| Classes (C) | 7 |
| Hidden layer size (H) | 16 |
| Total parameters | 23,063 (W1: 22,928 · b1: 16 · W2: 112 · b2: 7) |
| Train / Val / Test split | 140 / 500 / 1,000 nodes |
| Edge weights | Symmetric normalized adjacency D̃⁻¹/²ÃD̃⁻¹/² |

Features are stored as `float32` and downcast to the target format at runtime. All edge weights and node features enter the same 16-bit arithmetic pipeline as the weights.

---

## Project Structure

```
gcn-posit-experiment/
│
├── README.md                               ← This file
│
├── data/                                   ← Cora binary dataset
│   ├── features_fp32.bin                   #   Node feature matrix  [N × F]
│   ├── labels_int64.bin                    #   Class labels          [N]
│   ├── norm_edge_index_int64.bin           #   Edge src/dst indices  [2 × E]
│   ├── norm_edge_weight_fp32.bin           #   Normalized edge weights [E]
│   ├── train_mask_uint8.bin                #   Training node mask    [N]
│   ├── val_mask_uint8.bin                  #   Validation node mask  [N]
│   └── test_mask_uint8.bin                 #   Test node mask        [N]
│
├── checkpoints/                            ← Initial Xavier-initialized weights
│
├── universal/sw/                           ← Stillwater Universal Numbers library
│   └── universal/number/
│       ├── cfloat/cfloat.hpp               #   IEEE-style FP16 (cfloat<16,5>)
│       └── posit/posit.hpp                 #   Posit arithmetic (posit<16,1>)
│
└── experiment_200ep_3regimes/
    │
    ├── src/
    │   ├── cpp/
    │   │   ├── train_2layer_gd_200ep.cpp               ← Regime 1: Vanilla GD
    │   │   ├── train_2layer_pure16_adam_200ep.cpp       ← Regime 2: Pure 16-bit Adam
    │   │   ├── train_2layer_double_adam_200ep.cpp       ← Regime 3: Mixed Double Adam
    │   │   └── evaluate_3regimes_numerical_accuracy.cpp ← 80-bit audit engine
    │   │
    │   └── python/
    │       ├── build_dataset.py                ← Download and binarise Cora
    │       ├── generate_2layer_weights.py       ← Xavier weight initialisation
    │       ├── plot_all_regimes.py              ← Per-regime training curves
    │       ├── plot_numerical_accuracy_3regimes.py  ← Error bar charts
    │       └── plot_master_compiled_summary.py  ← Dashboard figure
    │
    └── results/
        ├── logs/
        │   └── numerical_accuracy_3regimes.csv  ← All audit metrics
        │
        ├── plots/
        │   ├── master_summary/                  ← Dashboard and compiled charts
        │   ├── regime1_without_optimizer/        ← Vanilla GD training curves
        │   ├── regime2_pure16_optimizer/         ← Pure Adam training curves
        │   └── regime3_double_optimizer/         ← Mixed Adam training curves
        │
        └── reports/
            ├── GCN_Posit_vs_FP16_Technical_Report.pdf
            ├── GCN_Posit_vs_FP_Complete_Defense_Guide.md
            ├── GCN_Posit_vs_FP_Complete_Defense_Guide.docx
            ├── GCN_Posit_Literature_Review_and_Citations.md
            └── GCN_Posit_Literature_Review_and_Citations.docx
```

---

## Quick Start

### Prerequisites

- **C++20** compiler (`g++` ≥ 10 or `clang++` ≥ 12)
- **Python 3.8+** with `numpy`, `matplotlib`, `torch-geometric` (for dataset build only)
- Stillwater Universal library (included under `universal/sw/`)

---

### Step 1 — Prepare the Dataset

```bash
cd experiment_200ep_3regimes
python3 src/python/build_dataset.py
python3 src/python/generate_2layer_weights.py
```

This downloads Cora, binarises all arrays, and saves Xavier-initialized weights to `checkpoints/`.

---

### Step 2 — Compile the C++ Engines

```bash
cd experiment_200ep_3regimes

g++ -std=c++20 -O3 -I../universal/sw \
    src/cpp/train_2layer_gd_200ep.cpp \
    -o train_regime1_gd

g++ -std=c++20 -O3 -I../universal/sw \
    src/cpp/train_2layer_pure16_adam_200ep.cpp \
    -o train_regime2_pure16_adam

g++ -std=c++20 -O3 -I../universal/sw \
    src/cpp/train_2layer_double_adam_200ep.cpp \
    -o train_regime3_double_adam

g++ -std=c++20 -O3 -I../universal/sw \
    src/cpp/evaluate_3regimes_numerical_accuracy.cpp \
    -o evaluate_3regimes
```

---

### Step 3 — Run All Experiments

```bash
./train_regime1_gd
./train_regime2_pure16_adam
./train_regime3_double_adam
./evaluate_3regimes
```

Training logs are written to `results/logs/`. The audit produces `numerical_accuracy_3regimes.csv`.

---

### Step 4 — Generate Plots

```bash
python3 src/python/plot_all_regimes.py
python3 src/python/plot_numerical_accuracy_3regimes.py
python3 src/python/plot_master_compiled_summary.py
```

All figures are saved to `results/plots/`.

---

## Reports and Documentation

| Document | Description |
|----------|-------------|
| [`GCN_Posit_vs_FP16_Technical_Report.pdf`](experiment_200ep_3regimes/results/reports/GCN_Posit_vs_FP16_Technical_Report.pdf) | Full technical report with all experimental results |
| [`GCN_Posit_vs_FP_Complete_Defense_Guide.md`](experiment_200ep_3regimes/results/reports/GCN_Posit_vs_FP_Complete_Defense_Guide.md) | Viva/thesis defense Q&A with mathematical derivations |
| [`GCN_Posit_vs_FP_Complete_Defense_Guide.docx`](experiment_200ep_3regimes/results/reports/GCN_Posit_vs_FP_Complete_Defense_Guide.docx) | Word version of the defense guide |
| [`GCN_Posit_Literature_Review_and_Citations.md`](experiment_200ep_3regimes/results/reports/GCN_Posit_Literature_Review_and_Citations.md) | Literature review and academic citations |
| [`FP16_vs_Posit16_Four_Failure_Modes.docx`](FP16_vs_Posit16_Four_Failure_Modes.docx) | Deep-dive: every formula behind all four FP16 failure modes |

---

## Implementation Notes

- **No PyTorch or ML framework** is used. Every forward pass, backward pass, and optimizer step is hand-coded in C++ using only the Stillwater Universal number types.
- **Dropout** uses inverted dropout (scale = 2.0, `p = 0.5`, seed = 42) — identical mask for FP16 and Posit16 ensuring a fair comparison.
- **Weight decay** (`5×10⁻⁴`) is applied only to `W1` to match standard GCN training practice.
- **Softmax and loss** are computed in `double` to avoid format-induced instability in the probability computation itself — only the logits going in are 16-bit.
- **Ground truth** for the audit is `long double` (80-bit extended precision, x86), giving ~18 significant decimal digits — over 10¹¹× more precise than the errors being measured.

---

## Number Library

This experiment uses the **Stillwater Universal Numbers** library:

```cpp
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>

using FP16_Type    = sw::universal::cfloat<16, 5>;   // IEEE-style half-precision
using Posit16_Type = sw::universal::posit<16, 1>;    // Posit with es=1
```

Both types are drop-in replacements for `float` — the entire GCN is a single template `<typename T>` that runs identically for both formats.

---

*Experiment conducted using Stillwater Universal Numbers library · Cora dataset · 200 training epochs · RNG seed 42*
