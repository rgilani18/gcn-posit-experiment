# Canonical GCN Posit Experiment Snapshot

This directory is a **non-destructive curated snapshot** of currently identified experiment artifacts. Original repository files remain untouched in their original locations.

## Model and dataset scope

- Model: 50%-pruned Cora GCN
- Architecture: 1433 -> 16 -> 7
- Dataset: Cora citation network
- Nodes: 2708
- Classes: 7

## Experiment split

A. **Independently trained model-accuracy comparison**
- Compare separately trained runs by format/variant (e.g., FP32, Posit32, FP16+double-Adam, Posit16) using available training/evaluation logs and plots.

B. **Same-checkpoint numerical comparison against long-double reference**
- Compare format behavior against long-double reference implementations and numerical-comparison utilities using shared inputs/checkpoint contexts.

## Important caveats

- Current curated artifacts **do not yet prove complete epoch-wise forward/backward numerical accuracy**.
- Per-epoch checkpoints and per-epoch backward gradient dumps are still required.
- A long-double backward evaluator is still required.
- `golden_logits_fp32.bin` is a PyTorch FP32 baseline artifact and **not** the final long-double golden reference.
- FP16 original and FP16 + double-Adam are separate variants and must not be conflated.
- Posit32 Adam remains a **candidate** until the exact source snapshot used for its result folder is fully verified.

## Canonical status table

| Format/variant | Source | Checkpoint/result | Status | Known limitations |
|---|---|---|---|---|
| FP32 Adam (500ep) | `gcn_train_fp32_adam.cpp`, `gcn_universal_fp32.cpp` | `results/fp32_adam_500ep/` | Candidate canonical (trained comparison) | Requires full environment/hash verification on magnum; does not alone prove epoch-wise numerical parity |
| Posit32 Adam (500ep) | `gcn_train_posit32_adam.cpp` | `results/posit32_adam_500ep/` | Candidate only | Source snapshot in result folder must be verified before treating as final |
| FP16 + double Adam | `gcn_train_fp16_adamstate_double.cpp`, `gcn_evaluate_fp16_adamdouble.cpp` | `results/corrected_testing/fp16_adamdouble.txt`, `gcn_trained_weights_fp16_adamdouble.bin` | Canonical candidate (16-bit corrected testing track) | Separate from FP16 original; no epoch-wise gradient dump evidence |
| Posit16 | `gcn_train_posit16_adam.cpp`, `gcn_evaluate_posit16.cpp` | `results/corrected_testing/posit16.txt`, `gcn_trained_weights_posit16_100ep.bin` | Canonical candidate (16-bit corrected testing track) | No epoch-wise backward verification against long-double |
| Long-double reference | `gcn_longdouble.cpp`, `gcn_longdouble_trained.cpp` | Source-level reference | Reference source available | Backward long-double evaluator not implemented |
| Partial numerical comparison utilities | `numerical_error_fp32_fp16_posit16.cpp`, `compare_reference.cpp`, `compare_numerical.py` | Utility outputs (see `manifest.csv`) | Partial evidence only | Not full epoch-wise forward/backward validation |

See `manifest.csv` and `NOTES.md` for detailed artifact-level status and publication blockers.
