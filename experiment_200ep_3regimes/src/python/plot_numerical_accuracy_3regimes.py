import csv
import subprocess
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

width, height = letter
px = 70
py = 110
panel_w = 470
panel_h = 280

def read_3regime_csv(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'regime': row['regime'],
                'format': row['format'],
                'test_acc': float(row['test_acc']),
                'val_loss': float(row['val_loss']),
                'mae': float(row['mae']),
                'rmse': float(row['rmse']),
                'max_err': float(row['max_err']),
                'disagreements': int(row['disagreements'])
            })
    return data

def plot_single_regime_accuracy(pdf_path, png_path, title, subtitle, fp16_row, pos16_row):
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title & Subtitle
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(px, height - 55, title)
    c.setFont("Helvetica", 9.5)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(px, height - 72, subtitle)

    # Background Box
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.rect(px, py, panel_w, panel_h, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(1)
    c.rect(px, py, panel_w, panel_h, fill=0, stroke=1)

    metrics = [
        ("Mean Absolute Error\n(MAE)", fp16_row['mae'], pos16_row['mae']),
        ("Root Mean Square Error\n(RMSE)", fp16_row['rmse'], pos16_row['rmse']),
        ("Maximum Absolute Error\n(Max Err)", fp16_row['max_err'], pos16_row['max_err'])
    ]

    group_w = panel_w / 3.0
    bar_w = 45
    bar_base_y = py + 35
    bar_max_h = 190

    for gi, (label, f_val, p_val) in enumerate(metrics):
        center_x = px + gi * group_w + group_w / 2.0
        max_val = max(f_val, p_val)
        scale = bar_max_h / (max_val * 1.35)

        f_h = max(f_val * scale, 3.0)
        p_h = max(p_val * scale, 3.0)

        # FP16 Bar (Red)
        f_x = center_x - bar_w - 6
        c.setFillColor(colors.HexColor("#ef4444"))
        c.rect(f_x, bar_base_y, bar_w, f_h, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#dc2626"))
        c.setLineWidth(1)
        c.rect(f_x, bar_base_y, bar_w, f_h, fill=0, stroke=1)

        # Posit16 Bar (Blue)
        p_x = center_x + 6
        c.setFillColor(colors.HexColor("#2563eb"))
        c.rect(p_x, bar_base_y, bar_w, p_h, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#1d4ed8"))
        c.setLineWidth(1)
        c.rect(p_x, bar_base_y, bar_w, p_h, fill=0, stroke=1)

        # Numerical Values on top of bars
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(f_x + bar_w/2.0, bar_base_y + f_h + 5, f"{f_val:.2e}")
        c.drawCentredString(p_x + bar_w/2.0, bar_base_y + p_h + 5, f"{p_val:.2e}")

        # Ratio Comparison
        ratio = f_val / p_val if p_val > 0 else 0
        c.setFillColor(colors.HexColor("#059669") if ratio >= 1.0 else colors.HexColor("#dc2626"))
        c.setFont("Helvetica-Bold", 8.5)
        if ratio >= 1.0:
            c.drawCentredString(center_x, bar_base_y + max(f_h, p_h) + 20, f"FP16 is {ratio:.1f}x higher error")
        else:
            c.drawCentredString(center_x, bar_base_y + max(f_h, p_h) + 20, f"Posit16 is {1.0/ratio:.1f}x higher error")

        # Metric Labels below
        c.setFillColor(colors.HexColor("#334155"))
        c.setFont("Helvetica-Bold", 8.5)
        lines = label.split('\n')
        for li, line in enumerate(lines):
            c.drawCentredString(center_x, bar_base_y - 14 - li * 11, line)

    # Legend — Strictly placed BELOW the graph so it NEVER covers data
    leg_y = py - 55
    c.setFillColor(colors.HexColor("#ef4444"))
    c.rect(px + 80, leg_y, 16, 10, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(px + 102, leg_y + 1, "IEEE FP16 (cfloat<16,5>)")

    c.setFillColor(colors.HexColor("#2563eb"))
    c.rect(px + 290, leg_y, 16, 10, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(px + 312, leg_y + 1, "Posit16 (posit<16,1>)")

    c.save()
    base_png = png_path[:-4]
    subprocess.run(["pdftoppm", "-png", "-r", "150", pdf_path, base_png], check=True)
    subprocess.run(["mv", f"{base_png}-1.png", png_path], check=True)
    print(f"Generated: {png_path}")

def plot_master_numerical_accuracy(pdf_path, png_path, rows):
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title & Subtitle
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(px, height - 52, "Master Numerical Accuracy Comparison Across All 3 Regimes")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(px, height - 68, "Logits MAE vs 80-Bit Long Double Ground Truth (Converged Weights at Epoch 200, 2708 Nodes)")

    # Background Box
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.rect(px, py, panel_w, panel_h, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(1)
    c.rect(px, py, panel_w, panel_h, fill=0, stroke=1)

    regimes_data = [
        ("Regime 1\n(Vanilla GD)", rows[0], rows[1]),
        ("Regime 2\n(Pure 16-bit Adam)", rows[2], rows[3]),
        ("Regime 3\n(Double Adam)", rows[4], rows[5])
    ]

    group_w = panel_w / 3.0
    bar_w = 45
    bar_base_y = py + 40
    bar_max_h = 180

    for gi, (label, f_row, p_row) in enumerate(regimes_data):
        center_x = px + gi * group_w + group_w / 2.0
        f_mae = f_row['mae']
        p_mae = p_row['mae']
        max_val = max(f_mae, p_mae)
        scale = bar_max_h / (max_val * 1.35)

        f_h = max(f_mae * scale, 3.0)
        p_h = max(p_mae * scale, 3.0)

        # FP16 Bar (Red)
        f_x = center_x - bar_w - 6
        c.setFillColor(colors.HexColor("#ef4444"))
        c.rect(f_x, bar_base_y, bar_w, f_h, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#dc2626"))
        c.setLineWidth(1)
        c.rect(f_x, bar_base_y, bar_w, f_h, fill=0, stroke=1)

        # Posit16 Bar (Blue)
        p_x = center_x + 6
        c.setFillColor(colors.HexColor("#2563eb"))
        c.rect(p_x, bar_base_y, bar_w, p_h, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#1d4ed8"))
        c.setLineWidth(1)
        c.rect(p_x, bar_base_y, bar_w, p_h, fill=0, stroke=1)

        # Labels on top
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(f_x + bar_w/2.0, bar_base_y + f_h + 5, f"{f_mae:.2e}")
        c.drawCentredString(p_x + bar_w/2.0, bar_base_y + p_h + 5, f"{p_mae:.2e}")

        # Ratio on top
        ratio = f_mae / p_mae if p_mae > 0 else 0
        c.setFillColor(colors.HexColor("#059669") if ratio >= 1.0 else colors.HexColor("#dc2626"))
        c.setFont("Helvetica-Bold", 8.5)
        if ratio >= 1.0:
            c.drawCentredString(center_x, bar_base_y + max(f_h, p_h) + 18, f"FP16 {ratio:.1f}x higher MAE")
        else:
            c.drawCentredString(center_x, bar_base_y + max(f_h, p_h) + 18, f"Posit16 {1.0/ratio:.1f}x higher MAE")

        # Regime labels below
        c.setFillColor(colors.HexColor("#334155"))
        c.setFont("Helvetica-Bold", 9)
        lines = label.split('\n')
        for li, line in enumerate(lines):
            c.drawCentredString(center_x, bar_base_y - 15 - li * 12, line)

    # Legend — Strictly placed BELOW the graph
    leg_y = py - 55
    c.setFillColor(colors.HexColor("#ef4444"))
    c.rect(px + 80, leg_y, 16, 10, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(px + 102, leg_y + 1, "IEEE FP16 (cfloat<16,5>)")

    c.setFillColor(colors.HexColor("#2563eb"))
    c.rect(px + 290, leg_y, 16, 10, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(px + 312, leg_y + 1, "Posit16 (posit<16,1>)")

    c.save()
    base_png = png_path[:-4]
    subprocess.run(["pdftoppm", "-png", "-r", "150", pdf_path, base_png], check=True)
    subprocess.run(["mv", f"{base_png}-1.png", png_path], check=True)
    print(f"Generated Master Plot: {png_path}")

if __name__ == "__main__":
    rows = read_3regime_csv("results/logs/numerical_accuracy_3regimes.csv")

    # 1. Regime 1 Plot
    plot_single_regime_accuracy(
        "results/plots/regime1_without_optimizer/numerical_accuracy.pdf",
        "results/plots/regime1_without_optimizer/numerical_accuracy.png",
        "Regime 1 (Vanilla GD): Numerical Accuracy vs Long Double",
        "Evaluated on Converged Epoch 200 Weights (2708 Nodes × 7 Classes)",
        rows[0], rows[1]
    )

    # 2. Regime 2 Plot
    plot_single_regime_accuracy(
        "results/plots/regime2_pure16_optimizer/numerical_accuracy.pdf",
        "results/plots/regime2_pure16_optimizer/numerical_accuracy.png",
        "Regime 2 (Pure 16-Bit Adam): Numerical Accuracy vs Long Double",
        "Shows Catastrophic Underflow in FP16 vs Stability in Posit16 (Epoch 200)",
        rows[2], rows[3]
    )

    # 3. Regime 3 Plot
    plot_single_regime_accuracy(
        "results/plots/regime3_double_optimizer/numerical_accuracy.pdf",
        "results/plots/regime3_double_optimizer/numerical_accuracy.png",
        "Regime 3 (Double Adam States): Numerical Accuracy vs Long Double",
        "Evaluated with 64-Bit Optimizer Assistance on Both Models (Epoch 200)",
        rows[4], rows[5]
    )

    # 4. Master Comparison Across All 3 Regimes
    plot_master_numerical_accuracy(
        "results/plots/master_summary/numerical_accuracy_all_3regimes.pdf",
        "results/plots/master_summary/numerical_accuracy_all_3regimes.png",
        rows
    )
    print("All numerical accuracy plots generated successfully!")

