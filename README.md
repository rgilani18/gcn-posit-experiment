# Numerical Accuracy of Floating-Point vs. Posit Arithmetic in Graph Convolutional Networks (GCNs)

## Project Overview
This project systematically evaluates the numerical accuracy, error propagation, training dynamics, and classification performance of **IEEE-754 standard floating-point (FP32, FP16)** versus **Posit arithmetic (Posit32, Posit16)** on Graph Convolutional Networks.

Arithmetic emulation for Posit and parameterized float representations is powered by the `sw::universal` library.

### Key Datasets & Models
- **Primary Benchmark**: 2-layer Graph Convolutional Network (GCN) on the **Cora** citation network (2,708 nodes, 1,433 features, 7 classes).
- **Generalization Benchmark**: GraphSAGE model on the **Amazon Photo** dataset.

---

## Directory Organization

```text
gcn-posit-experiment/
├── src/
│   ├── python/                # Python model definition, PyG training, data & weight export
│   │   ├── gcn_model.py       # 2-layer PyTorch Geometric GCN definition
│   │   ├── train_gcn.py       # Baseline PyTorch FP32 training
│   │   ├── prune_gcn.py       # 50% magnitude unstructured pruning
│   │   ├── export_cora.py     # Cora dataset extraction
│   │   ├── export_cora_masks.py # Train/Val/Test mask exporter
│   │   ├── export_gcn_norm.py # Exact graph convolution normalization computation
│   │   ├── export_cpp_data.py # Binary .bin serialization for C++ ingestion
│   │   ├── export_golden.py   # Golden reference PyTorch FP32 logits
│   │   ├── export_weights.py  # Model parameter exporter
│   │   ├── compare_numerical.py # FP32 vs Posit32 vs Long-Double error analysis
│   │   ├── verify_cora.py     # Data validation & sanity checker
│   │   ├── train_graphsage_amazon.py # GraphSAGE Amazon Photo trainer
│   │   ├── export_graphsage_*.py     # GraphSAGE export scripts
│   │   └── compare_graphsage_logits.py# GraphSAGE verification
│   │
│   ├── cpp/                   # C++ high-performance inference & training engines
│   │   ├── 32bit/             # FP32 vs Posit<32,2> comparison
│   │   │   ├── gcn_fp32.cpp             # First-principles FP32 GCN inference
│   │   │   ├── gcn_posit32.cpp          # Posit<32,2> GCN inference
│   │   │   ├── gcn_longdouble.cpp       # High-precision reference logits generator
│   │   │   ├── gcn_train_fp32_adam.cpp  # FP32 GCN training with Adam optimizer
│   │   │   ├── gcn_train_posit32_adam.cpp # Posit<32,2> GCN training with Adam
│   │   │   ├── gcn_compare_inference.cpp# Side-by-side inference metric comparison
│   │   │   └── compare_reference.cpp    # Error calculation against golden reference
│   │   │
│   │   ├── 16bit/             # FP16 vs Posit<16,1> comparison (Core Research Focus)
│   │   │   ├── gcn_train_fp16_adam.cpp  # Pure FP16 Adam (demonstrates underflow/instability)
│   │   │   ├── gcn_train_fp16_adamstate_double.cpp # Stabilized FP16 (Double Adam states)
│   │   │   ├── gcn_train_posit16_100ep.cpp # Native Posit<16,1> Adam training
│   │   │   ├── gcn_evaluate_fp16.cpp    # FP16 model evaluation
│   │   │   ├── gcn_evaluate_posit16.cpp # Posit16 model evaluation
│   │   │   └── numerical_error_fp32_fp16_posit16.cpp # Layer-wise activation error analysis
│   │   │
│   │   ├── graphsage/         # Amazon Photo GraphSAGE C++ implementation
│   │   │   ├── graphsage_fp32.cpp
│   │   │   └── graphsage_posit32.cpp
│   │   │
│   │   └── diagnostics/       # Micro-benchmarks, optimizer diagnostics & unit tests
│   │       ├── adam_step_numerical_diagnostic.cpp # Single Adam step underflow diagnostic
│   │       ├── gcn_train_fp16_underflow_diag.cpp  # Gradient underflow tracking
│   │       ├── test_softmax_gradient.cpp
│   │       ├── test_posit32.cpp
│   │       └── dd_precision_check.cpp
│   │
│   └── plots/                 # Figure generation scripts
│       ├── plot_16bit_final_curves.py  # Publication curves (loss, accuracy, stability)
│       ├── plot_16bit_comparison.py
│       └── plot_all_comparisons.py
│
├── results/                   # Experimental outputs, training logs, and generated figures
│   ├── logs/                  # CSV logs of training dynamics & text reports
│   │   ├── training_log_fp16.csv
│   │   ├── training_log_fp16_adamdouble.csv
│   │   ├── training_log_posit16_100ep.csv
│   │   ├── numerical_error_fp32_fp16_posit16.csv
│   │   ├── test_results_fp16_original.txt
│   │   ├── test_results_fp16_adamdouble.txt
│   │   ├── test_results_posit16.txt
│   │   └── precision_audit.txt
│   ├── plots/                 # Rendered PNG comparison charts
│   └── final_fp32_posit32_adam_500ep/ # 500-epoch Adam comparison reports & PDFs
│
├── data/                      # Raw and pre-processed PyTorch Geometric datasets
└── *.bin                      # Working binary tensors (features, normalized edge indices, weights)
```

---

## Key Experimental Findings

1. **Inference Numerical Accuracy**:
   - Layer-by-layer forward activation profiling shows **Posit16 achieves ~2× lower Mean Absolute Error (MAE)** and **4×–8× lower maximum absolute error** compared to FP16 at identical 16-bit storage width.
   - When using FP32 pre-trained weights directly for 16-bit inference:
     - **FP16 inference**: Acc drops to **10.3%** (near random) due to exponent limitation and dynamic range clipping.
     - **Posit16 inference**: Maintains **79.4%** test accuracy due to the tapered accuracy property of the regime bits.

2. **Training & Optimizer Dynamics**:
   - **Pure FP16 Adam**: Suffers catastrophic divergence by epoch 2 (validation loss exploding to 33.18) due to underflow in the second-moment ($v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$) updates.
   - **FP16 + Double Adam States**: Recovers stability, converging in 55 epochs to **76.4%** validation accuracy and **78.1%** test accuracy.
   - **Native Posit16 Adam**: Converges smoothly without special moment widening in 51 epochs, achieving **75.6%** validation accuracy and **79.4%** test accuracy.

