import numpy as np

NUM_NODES = 2708
NUM_CLASSES = 7
N = NUM_NODES * NUM_CLASSES

# -----------------------------
# Load data
# -----------------------------

fp32 = np.fromfile(
    "logits_fp32.bin",
    dtype=np.float32,
    count=N
)

posit32 = np.fromfile(
    "logits_posit32.bin",
    dtype=np.float32,
    count=N
)

longdouble = np.fromfile(
    "golden_logits_longdouble.bin",
    dtype=np.longdouble,
    count=N
)

# Test mask
test_mask = np.fromfile(
    "test_mask_uint8.bin",
    dtype=np.uint8,
    count=NUM_NODES
).astype(bool)

# -----------------------------
# Reshape logits
# -----------------------------

fp32 = fp32.reshape(NUM_NODES, NUM_CLASSES)
posit32 = posit32.reshape(NUM_NODES, NUM_CLASSES)
longdouble = longdouble.reshape(NUM_NODES, NUM_CLASSES)

# Select only test nodes
fp32_test = fp32[test_mask].astype(np.longdouble)
posit32_test = posit32[test_mask].astype(np.longdouble)
longdouble_test = longdouble[test_mask]

print("\n===== Numerical Accuracy Comparison =====")
print("Test nodes:", np.sum(test_mask))

# -----------------------------
# Numerical error
# -----------------------------

fp32_error = np.abs(fp32_test - longdouble_test)
posit_error = np.abs(posit32_test - longdouble_test)

# Relative error
denominator = np.maximum(
    np.abs(longdouble_test),
    np.longdouble(1e-30)
)

fp32_relative = fp32_error / denominator
posit_relative = posit_error / denominator

# -----------------------------
# Results
# -----------------------------

print("\nFP32:")
print("  Max absolute error :", np.max(fp32_error))
print("  Mean absolute error:", np.mean(fp32_error))
print("  Max relative error :", np.max(fp32_relative))
print("  Mean relative error:", np.mean(fp32_relative))

print("\nPosit32:")
print("  Max absolute error :", np.max(posit_error))
print("  Mean absolute error:", np.mean(posit_error))
print("  Max relative error :", np.max(posit_relative))
print("  Mean relative error:", np.mean(posit_relative))

# -----------------------------
# Error ratios
# -----------------------------

print("\n===== Error Ratio =====")

print(
    "Max absolute error ratio (FP32 / Posit32):",
    np.max(fp32_error) / np.max(posit_error)
)

print(
    "Mean absolute error ratio (FP32 / Posit32):",
    np.mean(fp32_error) / np.mean(posit_error)
)

# -----------------------------
# Prediction agreement
# -----------------------------

fp32_pred = np.argmax(fp32[test_mask], axis=1)
posit_pred = np.argmax(posit32[test_mask], axis=1)
golden_pred = np.argmax(longdouble[test_mask], axis=1)

fp32_agreement = np.mean(fp32_pred == golden_pred)
posit_agreement = np.mean(posit_pred == golden_pred)

print("\n===== Prediction Agreement with Reference =====")

print(
    "FP32 agreement with reference:",
    fp32_agreement
)

print(
    "Posit32 agreement with reference:",
    posit_agreement
)
