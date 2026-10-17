# Canonical data expectations

This snapshot does **not** duplicate large binary inputs by default. Canonical runs are expected to reference existing repository inputs in-place.

Expected canonical input files and sizes:

| File | Expected size (bytes) |
|---|---:|
| `features_fp32.bin` | 15522256 |
| `norm_edge_index_int64.bin` | 212224 |
| `norm_edge_weight_fp32.bin` | 53056 |
| `labels_int64.bin` | 21664 |
| `train_mask_uint8.bin` | 2708 |
| `val_mask_uint8.bin` | 2708 |
| `test_mask_uint8.bin` | 2708 |

Data/checkpoint hashes and actual server execution provenance still need verification on **magnum** before final publication.
