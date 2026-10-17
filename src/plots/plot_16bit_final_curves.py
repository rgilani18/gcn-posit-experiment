import pandas as pd
import matplotlib.pyplot as plt

runs = {
    "Original FP16": "training_log_fp16_100ep.csv",
    "FP16 + double Adam": "training_log_fp16_adamdouble.csv",
    "Posit16": "training_log_posit16_100ep.csv",
}

data = {}
for name, filename in runs.items():
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    data[name] = df

def save_plot(series, ylabel, title, filename, log_scale=False):
    plt.figure(figsize=(9, 5.5))
    for name, df, column in series:
        plt.plot(df["epoch"], df[column], label=name, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if log_scale:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()
    print("Saved:", filename)

save_plot(
    [(n, d, "val_accuracy_after_update") for n, d in data.items()],
    "Validation accuracy", "Validation Accuracy vs Epoch",
    "final_validation_accuracy_16bit.png"
)

save_plot(
    [(n, data[n], "val_loss_after_update")
     for n in ["FP16 + double Adam", "Posit16"]],
    "Validation cross-entropy", "Validation Loss vs Epoch: Stable Runs",
    "final_validation_loss_stable_16bit.png"
)

save_plot(
    [("Original FP16", data["Original FP16"], "val_loss_after_update")],
    "Validation cross-entropy", "Original FP16: Validation Loss vs Epoch",
    "final_validation_loss_original_fp16.png"
)

save_plot(
    [(n, d, "train_loss_before_update") for n, d in data.items()],
    "Training cross-entropy", "Training Loss vs Epoch",
    "final_training_loss_16bit.png"
)
