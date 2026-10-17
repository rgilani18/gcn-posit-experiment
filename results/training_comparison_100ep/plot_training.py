import pandas as pd
import matplotlib.pyplot as plt

fp32 = pd.read_csv("training_log_fp32.csv")
posit = pd.read_csv("training_log_posit32.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(fp32["epoch"], fp32["loss_before_update"],
             label="FP32", linewidth=2)
axes[0].plot(posit["epoch"], posit["loss_before_update"],
             label="Posit32", linewidth=2)
axes[0].set_title("Training Loss vs Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(fp32["epoch"], fp32["train_accuracy_before_update"],
             label="FP32", linewidth=2)
axes[1].plot(posit["epoch"], posit["train_accuracy_before_update"],
             label="Posit32", linewidth=2)
axes[1].set_title("Training Accuracy vs Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Training Accuracy")
axes[1].set_ylim(0, 1.05)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("training_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: training_comparison.png")
print("FP32 epochs:", len(fp32))
print("Posit32 epochs:", len(posit))
print("Training accuracy differs at epochs:",
      (fp32["train_accuracy_before_update"] !=
       posit["train_accuracy_before_update"]).sum())
print("Maximum logged loss difference:",
      (fp32["loss_before_update"] -
       posit["loss_before_update"]).abs().max())
