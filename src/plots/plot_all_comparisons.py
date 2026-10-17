import glob
import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = "results/final_comparison_graphs"
os.makedirs(out, exist_ok=True)

logs = {}
for file in sorted(glob.glob("training_log*.csv")):
    try:
        df = pd.read_csv(file)
        df.columns = [c.strip().lower() for c in df.columns]
        if "epoch" in df.columns and len(df):
            name = os.path.basename(file).replace(".csv", "")
            logs[name] = df
    except Exception as e:
        print(f"Skipped {file}: {e}")

if not logs:
    raise SystemExit("No readable training_log*.csv files found.")

print("Logs included:")
for name, df in logs.items():
    print(f"  {name}: {len(df)} epochs")

metrics = {
    "training_loss": ["train_loss"],
    "validation_loss": ["val_loss"],
    "training_accuracy": ["train_accuracy"],
    "validation_accuracy": ["val_accuracy"],
}

summary = []

for name, df in logs.items():
    row = {"run": name, "epochs_logged": len(df)}
    for metric, patterns in metrics.items():
        cols = [c for c in df.columns
                if any(p in c for p in patterns)]
        if cols:
            row[metric + "_final"] = df[cols[-1]].iloc[-1]
    summary.append(row)

for metric, patterns in metrics.items():
    plt.figure(figsize=(10, 6))
    count = 0

    for name, df in logs.items():
        cols = [c for c in df.columns
                if any(p in c for p in patterns)]
        if not cols:
            continue

        col = cols[-1]
        y = pd.to_numeric(df[col], errors="coerce")
        valid = y.notna()
        if not valid.any():
            continue

        plt.plot(df.loc[valid, "epoch"], y[valid],
                 label=name, linewidth=1.8)
        count += 1

    if count == 0:
        plt.close()
        continue

    if "loss" in metric:
        plt.yscale("symlog", linthresh=0.1)
        plt.ylabel("Loss (symlog scale)")
    else:
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.02)

    plt.xlabel("Epoch")
    plt.title(metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = f"{out}/{metric}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print("Saved:", path)

pd.DataFrame(summary).to_csv(
    f"{out}/final_metrics_summary.csv", index=False
)
print("Saved:", f"{out}/final_metrics_summary.csv")
print("Done. Review which runs were included before reporting.")
