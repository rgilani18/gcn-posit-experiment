import csv
import matplotlib.pyplot as plt

files = {
    "FP16 + double Adam states": "training_log_fp16_adamdouble.csv",
    "Posit16": "training_log_posit16_100ep.csv",
}

logs = {}
for label, filename in files.items():
    with open(filename, newline="") as f:
        logs[label] = list(csv.DictReader(f))

# Validation loss
plt.figure()
for label, rows in logs.items():
    epochs = [int(r["epoch"]) for r in rows]
    values = [float(r["val_loss_after_update"]) for r in rows]
    plt.plot(epochs, values, label=label)
plt.xlabel("Epoch")
plt.ylabel("Validation Cross-Entropy Loss")
plt.title("16-bit GCN: Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_16bit_validation_loss.png", dpi=200)
plt.close()

# Validation accuracy
plt.figure()
for label, rows in logs.items():
    epochs = [int(r["epoch"]) for r in rows]
    values = [100 * float(r["val_accuracy_after_update"]) for r in rows]
    plt.plot(epochs, values, label=label)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("16-bit GCN: Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_16bit_validation_accuracy.png", dpi=200)
plt.close()

print("Created:")
print("  comparison_16bit_validation_loss.png")
print("  comparison_16bit_validation_accuracy.png")
