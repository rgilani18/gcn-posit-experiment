import numpy as np

N = 2708   # Nodes
F = 1433   # Input Features
H = 16     # Hidden Units
C = 7      # Classes

SEED = 2026
np.random.seed(SEED)

print(f"=== Generating 2-Layer Initial Weights (Seed={SEED}) ===")

std1 = np.sqrt(2.0 / F)
std2 = np.sqrt(2.0 / H)

# Layer 1: W1 (H x F), b1 (H)
w1 = np.random.normal(0.0, std1, size=(H * F)).astype(np.float32)
b1 = np.zeros(H, dtype=np.float32)

# Layer 2: W2 (C x H), b2 (C)
w2 = np.random.normal(0.0, std2, size=(C * H)).astype(np.float32)
b2 = np.zeros(C, dtype=np.float32)

output_path = "checkpoints/gcn_initial_weights_2layer_seed2026.bin"

with open(output_path, "wb") as f:
    f.write(w1.tobytes())
    f.write(b1.tobytes())
    f.write(w2.tobytes())
    f.write(b2.tobytes())

total_floats = len(w1) + len(b1) + len(w2) + len(b2)
print(f"  Layer 1 (W1: {H}x{F}, b1: {H}): {len(w1) + len(b1)} parameters")
print(f"  Layer 2 (W2: {C}x{H}, b2: {C}): {len(w2) + len(b2)} parameters")
print(f"  Total Parameters: {total_floats} float32 values ({total_floats * 4} bytes)")
print(f"  Saved to: {output_path}")

