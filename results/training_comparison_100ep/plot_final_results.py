import csv
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fp = list(csv.DictReader(open("training_log_fp32.csv")))
po = list(csv.DictReader(open("training_log_posit32.csv")))

# 1. Training loss
plt.figure(figsize=(9, 5))
for rows, name in [(fp, "FP32"), (po, "Posit32")]:
    plt.plot([int(r["epoch"]) for r in rows],
             [float(r["loss_before_update"]) for r in rows], label=name)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("GCN Training Loss: FP32 vs Posit32")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss_comparison.png", dpi=200)
plt.close()

# 2. Training accuracy
plt.figure(figsize=(9, 5))
for rows, name in [(fp, "FP32"), (po, "Posit32")]:
    plt.plot([int(r["epoch"]) for r in rows],
             [100*float(r["train_accuracy_before_update"]) for r in rows],
             label=name)
plt.xlabel("Epoch")
plt.ylabel("Training accuracy (%)")
plt.title("GCN Training Accuracy: FP32 vs Posit32")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_accuracy_comparison.png", dpi=200)
plt.close()

# Parse all-node RMSE from each inference report
reports = [
    ("FP32-trained checkpoint", "inference_fp32_trained_weights.txt"),
    ("Posit32-trained checkpoint", "inference_posit32_trained_weights.txt"),
]
rmse = {"FP32": [], "Posit32": []}
test_acc = {"FP32": [], "Posit32": []}

for label, filename in reports:
    text = Path(filename).read_text()

    for fmt, prefix in [
        ("FP32", "FP32 all nodes"),
        ("Posit32", "Posit32 all nodes"),
    ]:
        m = re.search(re.escape(prefix) + r".*?RMSE=([0-9.eE+-]+)", text)
        if not m:
            raise ValueError(f"Missing RMSE: {prefix} in {filename}")
        rmse[fmt].append(float(m.group(1)))

    sections = re.split(r"--- Posit32 metrics ---", text, maxsplit=1)
    if len(sections) != 2:
        raise ValueError(f"Could not parse metrics in {filename}")
    for fmt, section in zip(["FP32", "Posit32"], sections):
        m = re.search(r"Test: accuracy=([0-9.]+)", section)
        if not m:
            raise ValueError(f"Missing test accuracy in {filename}")
        test_acc[fmt].append(100*float(m.group(1)))

# 3. Numerical RMSE (log scale)
x = [0, 1]
w = 0.34
plt.figure(figsize=(9, 5))
plt.bar([i-w/2 for i in x], rmse["FP32"], w, label="FP32")
plt.bar([i+w/2 for i in x], rmse["Posit32"], w, label="Posit32")
plt.xticks(x, [r[0] for r in reports])
plt.ylabel("All-node logit RMSE (log scale)")
plt.yscale("log")
plt.title("Numerical Error Against Double-Precision Reference")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("inference_logit_rmse.png", dpi=200)
plt.close()

# 4. Test accuracy
plt.figure(figsize=(9, 5))
plt.bar([i-w/2 for i in x], test_acc["FP32"], w, label="FP32")
plt.bar([i+w/2 for i in x], test_acc["Posit32"], w, label="Posit32")
plt.xticks(x, [r[0] for r in reports])
plt.ylabel("Test accuracy (%)")
plt.ylim(0, 100)
plt.title("Test Accuracy by Checkpoint and Arithmetic")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("test_accuracy_comparison.png", dpi=200)
plt.close()

print("Generated:")
for name in [
    "training_loss_comparison.png",
    "training_accuracy_comparison.png",
    "inference_logit_rmse.png",
    "test_accuracy_comparison.png",
]:
    p = Path(name)
    print(f"{name}: {p.stat().st_size:,} bytes")
