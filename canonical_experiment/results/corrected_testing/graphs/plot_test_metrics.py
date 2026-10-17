import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("results/corrected_testing/graphs")
models = ["FP16 original", "FP16 + double Adam states", "Posit16"]

test_accuracy = [10.3, 78.1, 79.4]
test_cross_entropy = [8.51776, 0.768494, 0.689849]

# Graph 1: Test accuracy
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, test_accuracy)
ax.set_title("GCN Test Accuracy Comparison")
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)
ax.bar_label(bars, fmt="%.1f%%", padding=3)
plt.xticks(rotation=12, ha="right")
fig.tight_layout()
fig.savefig(out / "test_accuracy_corrected.png", dpi=300)
plt.close(fig)

# Graph 2: Test cross-entropy
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, test_cross_entropy)
ax.set_title("GCN Test Cross-Entropy Comparison")
ax.set_ylabel("Test cross-entropy (lower is better)")
ax.grid(axis="y", alpha=0.3)
ax.bar_label(bars, fmt="%.4f", padding=3)
plt.xticks(rotation=12, ha="right")
fig.tight_layout()
fig.savefig(out / "test_cross_entropy_corrected.png", dpi=300)
plt.close(fig)

print("Saved:")
print(out / "test_accuracy_corrected.png")
print(out / "test_cross_entropy_corrected.png")
