import csv
import subprocess
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

width, height = letter
panel_w = 460
panel_h = 280
px = 75
py = 110

def read_csv(filename):
    epochs, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = [], [], [], [], [], [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            train_acc.append(float(row['train_acc']))
            val_loss.append(float(row['val_loss']))
            val_acc.append(float(row['val_acc']))
            test_loss.append(float(row['test_loss']))
            test_acc.append(float(row['test_acc']))
    return {
        'epoch': epochs, 'train_loss': train_loss, 'train_acc': train_acc,
        'val_loss': val_loss, 'val_acc': val_acc, 'test_loss': test_loss, 'test_acc': test_acc
    }

def map_x_200(ep):
    return px + (ep - 1) / 199.0 * panel_w

def generate_2curve_graph(
    pdf_path, png_path, title, y_label,
    y_ticks, max_y,
    fp16_data, pos16_data,
    fp16_label, pos16_label,
    show_random_guess=False
):
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(px, height - 60, title)

    # Background Box
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.rect(px, py, panel_w, panel_h, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(1)
    c.rect(px, py, panel_w, panel_h, fill=0, stroke=1)

    # Y-axis Gridlines and labels
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#64748b"))
    for y_val in y_ticks:
        y_pt = py + (y_val / float(max_y)) * panel_h
        c.setStrokeColor(colors.HexColor("#e2e8f0"))
        c.setLineWidth(0.8)
        c.line(px, y_pt, px + panel_w, y_pt)
        c.drawString(px - 38, y_pt - 3, f"{y_val:.1f}" if isinstance(y_val, float) else f"{y_val}%")

    # X-axis ticks (1, 40, 80, 120, 160, 200)
    for ep in [1, 40, 80, 120, 160, 200]:
        x_pt = map_x_200(ep)
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.line(x_pt, py, x_pt, py - 5)
        c.drawString(x_pt - 9, py - 18, str(ep))

    # Axis Labels
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.HexColor("#334155"))
    c.drawString(px + panel_w/2 - 20, py - 35, "Epoch")
    
    c.saveState()
    c.translate(px - 45, py + panel_h/2)
    c.rotate(90)
    c.drawString(-len(y_label)*3, 0, y_label)
    c.restoreState()

    if show_random_guess:
        y_rand = py + (14.28 / float(max_y)) * panel_h
        c.setStrokeColor(colors.HexColor("#64748b"))
        c.setLineWidth(1.2)
        c.setDash([3, 3], 0)
        c.line(px, y_rand, px + panel_w, y_rand)
        c.drawString(px + panel_w - 115, y_rand + 4, "Random Guess (14.28%)")
        c.setDash([], 0)

    # Draw Curves with Canvas Clipping
    c.saveState()
    clip_p = c.beginPath()
    clip_p.rect(px, py, panel_w, panel_h)
    c.clipPath(clip_p, stroke=0)

    def map_y(val):
        return py + (val / float(max_y)) * panel_h

    # Draw FP16 Line (Red Dashed)
    c.setStrokeColor(colors.HexColor("#ef4444"))
    c.setLineWidth(2.2)
    c.setDash([5, 3], 0)
    p = c.beginPath()
    p.moveTo(map_x_200(1), map_y(fp16_data[0]))
    for ep, y in enumerate(fp16_data[1:], start=2):
        p.lineTo(map_x_200(ep), map_y(y))
    c.drawPath(p)

    # Draw Posit16 Line (Blue Solid)
    c.setStrokeColor(colors.HexColor("#2563eb"))
    c.setLineWidth(2.5)
    c.setDash([], 0)
    p = c.beginPath()
    p.moveTo(map_x_200(1), map_y(pos16_data[0]))
    for ep, y in enumerate(pos16_data[1:], start=2):
        p.lineTo(map_x_200(ep), map_y(y))
    c.drawPath(p)

    c.restoreState()

    # Legend — placed BELOW the graph area so it never covers data
    leg_y = py - 62
    leg_x_start = px + 60

    # FP16 Legend Entry (left side)
    c.setStrokeColor(colors.HexColor("#ef4444"))
    c.setLineWidth(2.2)
    c.setDash([5, 3], 0)
    c.line(leg_x_start, leg_y + 5, leg_x_start + 25, leg_y + 5)
    c.setDash([], 0)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(leg_x_start + 30, leg_y + 2, fp16_label)

    # Posit16 Legend Entry (right side)
    leg_x2 = leg_x_start + 180
    c.setStrokeColor(colors.HexColor("#2563eb"))
    c.setLineWidth(2.5)
    c.setDash([], 0)
    c.line(leg_x2, leg_y + 5, leg_x2 + 25, leg_y + 5)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(leg_x2 + 30, leg_y + 2, pos16_label)

    c.save()
    base_png = png_path[:-4]
    subprocess.run(["pdftoppm", "-png", "-r", "150", pdf_path, base_png], check=True)
    subprocess.run(["mv", f"{base_png}-1.png", png_path], check=True)
    print(f"Generated: {png_path}")

def plot_regime(folder, fp16_file, pos16_file, regime_name, loss_max):
    fp16 = read_csv(fp16_file)
    pos16 = read_csv(pos16_file)

    loss_ticks = [0.0, loss_max * 0.2, loss_max * 0.4, loss_max * 0.6, loss_max * 0.8, float(loss_max)]

    # 1. Validation Loss
    generate_2curve_graph(
        f"{folder}/1_validation_loss.pdf", f"{folder}/1_validation_loss.png",
        f"{regime_name}: Validation Loss (200 Epochs)", "Validation Loss",
        loss_ticks, loss_max,
        fp16['val_loss'], pos16['val_loss'],
        "FP16 Val Loss", "Posit16 Val Loss"
    )

    # 2. Test Loss
    generate_2curve_graph(
        f"{folder}/2_test_loss.pdf", f"{folder}/2_test_loss.png",
        f"{regime_name}: Test Loss (200 Epochs)", "Test Loss",
        loss_ticks, loss_max,
        fp16['test_loss'], pos16['test_loss'],
        "FP16 Test Loss", "Posit16 Test Loss"
    )

    # 3. Validation Accuracy
    generate_2curve_graph(
        f"{folder}/3_validation_accuracy.pdf", f"{folder}/3_validation_accuracy.png",
        f"{regime_name}: Validation Accuracy (%) (200 Epochs)", "Validation Accuracy (%)",
        [0, 20, 40, 60, 80, 100], 100,
        fp16['val_acc'], pos16['val_acc'],
        "FP16 Val Acc", "Posit16 Val Acc",
        show_random_guess=True
    )

    # 4. Test Accuracy
    generate_2curve_graph(
        f"{folder}/4_test_accuracy.pdf", f"{folder}/4_test_accuracy.png",
        f"{regime_name}: Test Accuracy (%) (200 Epochs)", "Test Accuracy (%)",
        [0, 20, 40, 60, 80, 100], 100,
        fp16['test_acc'], pos16['test_acc'],
        "FP16 Test Acc", "Posit16 Test Acc",
        show_random_guess=True
    )

def plot_master_summary():
    gd_fp16 = read_csv("results/logs/training_log_200ep_fp16_gd.csv")
    gd_pos = read_csv("results/logs/training_log_200ep_posit16_gd.csv")
    a16_fp16 = read_csv("results/logs/training_log_200ep_fp16_adam16.csv")
    a16_pos = read_csv("results/logs/training_log_200ep_posit16_adam16.csv")
    ad_fp16 = read_csv("results/logs/training_log_200ep_fp16_adam_double.csv")
    ad_pos = read_csv("results/logs/training_log_200ep_posit16_adam_double.csv")

    pdf_path = "results/plots/master_summary/master_comparison_all_regimes.pdf"
    png_path = "results/plots/master_summary/master_comparison_all_regimes.png"

    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(px, height - 55, "Master Comparison: Test Accuracy Across All 3 Regimes (200 Epochs)")

    # Background Box
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.rect(px, py, panel_w, panel_h, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(1)
    c.rect(px, py, panel_w, panel_h, fill=0, stroke=1)

    # Y-axis ticks
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#64748b"))
    for y_val in [0, 20, 40, 60, 80, 100]:
        y_pt = py + (y_val / 100.0) * panel_h
        c.setStrokeColor(colors.HexColor("#e2e8f0"))
        c.setLineWidth(0.8)
        c.line(px, y_pt, px + panel_w, y_pt)
        c.drawString(px - 34, y_pt - 3, f"{y_val}%")

    # X-axis ticks
    for ep in [1, 40, 80, 120, 160, 200]:
        x_pt = map_x_200(ep)
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.line(x_pt, py, x_pt, py - 5)
        c.drawString(x_pt - 9, py - 18, str(ep))

    # Axis Labels
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.HexColor("#334155"))
    c.drawString(px + panel_w/2 - 20, py - 35, "Epoch")
    
    c.saveState()
    c.translate(px - 45, py + panel_h/2)
    c.rotate(90)
    c.drawString(-len("Test Accuracy (%)")*3, 0, "Test Accuracy (%)")
    c.restoreState()

    # Random Guess Line
    y_rand = py + (14.28 / 100.0) * panel_h
    c.setStrokeColor(colors.HexColor("#64748b"))
    c.setLineWidth(1.0)
    c.setDash([3, 3], 0)
    c.line(px, y_rand, px + panel_w, y_rand)
    c.drawString(px + panel_w - 115, y_rand + 4, "Random Guess (14.28%)")
    c.setDash([], 0)

    # Curves to draw:
    # 1. GD FP16: Orange dashed
    # 2. GD Posit16: Orange solid
    # 3. Adam16 FP16: Red dashed
    # 4. Adam16 Posit16: Blue solid
    # 5. Double Adam FP16: Purple dashed
    # 6. Double Adam Posit16: Green/Teal solid

    curves = [
        (gd_fp16['test_acc'], "#f97316", [4, 3], 1.8, "FP16 (No Optimizer)"),
        (gd_pos['test_acc'], "#ea580c", [], 2.2, "Posit16 (No Optimizer)"),
        (a16_fp16['test_acc'], "#ef4444", [5, 3], 2.2, "FP16 (Pure 16-bit Adam)"),
        (a16_pos['test_acc'], "#2563eb", [], 2.5, "Posit16 (Pure 16-bit Adam)"),
        (ad_fp16['test_acc'], "#9333ea", [4, 2], 2.0, "FP16 (Double Adam)"),
        (ad_pos['test_acc'], "#059669", [], 2.2, "Posit16 (Double Adam)")
    ]

    c.saveState()
    clip_p = c.beginPath()
    clip_p.rect(px, py, panel_w, panel_h)
    c.clipPath(clip_p, stroke=0)

    def map_y(val):
        return py + (val / 100.0) * panel_h

    for data, color_hex, dash, lw, label in curves:
        c.setStrokeColor(colors.HexColor(color_hex))
        c.setLineWidth(lw)
        c.setDash(dash, 0)
        p = c.beginPath()
        p.moveTo(map_x_200(1), map_y(data[0]))
        for ep, y in enumerate(data[1:], start=2):
            p.lineTo(map_x_200(ep), map_y(y))
        c.drawPath(p)

    c.restoreState()

    # Legend — placed BELOW the graph, 2 columns x 3 rows, so it never covers data
    leg_y_base = py - 55
    col1_x = px + 10
    col2_x = px + panel_w / 2 + 10

    for i, (data, color_hex, dash, lw, label) in enumerate(curves):
        col = i % 2
        row = i // 2
        lx = col1_x if col == 0 else col2_x
        ly = leg_y_base - row * 16

        c.setStrokeColor(colors.HexColor(color_hex))
        c.setLineWidth(lw)
        c.setDash(dash, 0)
        c.line(lx, ly + 4, lx + 22, ly + 4)
        c.setDash([], 0)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 8)
        c.drawString(lx + 27, ly, f"{label}: {data[-1]:.1f}%")

    c.save()
    base_png = png_path[:-4]
    subprocess.run(["pdftoppm", "-png", "-r", "150", pdf_path, base_png], check=True)
    subprocess.run(["mv", f"{base_png}-1.png", png_path], check=True)
    print(f"Generated Master Plot: {png_path}")

def plot_numerical_accuracy():
    """Generate a numerical accuracy comparison bar chart (MAE, RMSE, Max Error)"""
    import csv as csv_mod
    import math

    # Read summary CSV
    with open("results/logs/numerical_accuracy_summary.csv") as f:
        reader = csv_mod.DictReader(f)
        rows = {r['format']: r for r in reader}

    fp16_mae = float(rows['FP16']['mae'])
    fp16_rmse = float(rows['FP16']['rmse'])
    fp16_max = float(rows['FP16']['max_error'])
    pos16_mae = float(rows['Posit16']['mae'])
    pos16_rmse = float(rows['Posit16']['rmse'])
    pos16_max = float(rows['Posit16']['max_error'])

    pdf_path = "results/plots/master_summary/numerical_accuracy_comparison.pdf"
    png_path = "results/plots/master_summary/numerical_accuracy_comparison.png"

    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(px, height - 55, "Numerical Accuracy: FP16 vs Posit16 vs Long Double")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(px, height - 72, "2-Layer GCN Forward Pass (Initial Weights, Seed 2026, 2708 Nodes × 7 Classes)")

    # Draw 3 grouped bar pairs: MAE, RMSE, Max Error
    metrics = [
        ("Mean Absolute\nError (MAE)", fp16_mae, pos16_mae),
        ("Root Mean Square\nError (RMSE)", fp16_rmse, pos16_rmse),
        ("Maximum\nAbsolute Error", fp16_max, pos16_max)
    ]

    bar_area_x = px + 30
    bar_area_w = panel_w - 60
    bar_area_y = py + 40
    bar_area_h = 220

    group_w = bar_area_w / 3
    bar_w = 45

    # Background
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.rect(px, py, panel_w, panel_h, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(1)
    c.rect(px, py, panel_w, panel_h, fill=0, stroke=1)

    for gi, (label, fp16_val, pos16_val) in enumerate(metrics):
        group_center = bar_area_x + gi * group_w + group_w / 2

        max_val = max(fp16_val, pos16_val)
        scale = bar_area_h / (max_val * 1.3)

        fp16_h = fp16_val * scale
        pos16_h = pos16_val * scale

        # FP16 bar (Red)
        fp16_x = group_center - bar_w - 5
        c.setFillColor(colors.HexColor("#ef4444"))
        c.rect(fp16_x, bar_area_y, bar_w, fp16_h, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#dc2626"))
        c.setLineWidth(1)
        c.rect(fp16_x, bar_area_y, bar_w, fp16_h, fill=0, stroke=1)

        # Posit16 bar (Blue)
        pos16_x = group_center + 5
        c.setFillColor(colors.HexColor("#2563eb"))
        c.rect(pos16_x, bar_area_y, bar_w, pos16_h, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#1d4ed8"))
        c.setLineWidth(1)
        c.rect(pos16_x, bar_area_y, bar_w, pos16_h, fill=0, stroke=1)

        # Value labels on top of bars
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(fp16_x + bar_w/2, bar_area_y + fp16_h + 5, f"{fp16_val:.2e}")
        c.drawCentredString(pos16_x + bar_w/2, bar_area_y + pos16_h + 5, f"{pos16_val:.2e}")

        # Ratio label
        ratio = fp16_val / pos16_val
        c.setFillColor(colors.HexColor("#059669"))
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(group_center, bar_area_y + max(fp16_h, pos16_h) + 22, f"FP16 is {ratio:.1f}× worse")

        # Metric label below
        c.setFillColor(colors.HexColor("#334155"))
        c.setFont("Helvetica-Bold", 9)
        lines = label.split('\n')
        for li, line in enumerate(lines):
            c.drawCentredString(group_center, bar_area_y - 18 - li * 12, line)

    # Legend below graph
    leg_y = py - 55
    c.setFillColor(colors.HexColor("#ef4444"))
    c.rect(px + 100, leg_y, 15, 10, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(px + 120, leg_y + 1, "IEEE FP16 (cfloat<16,5>)")

    c.setFillColor(colors.HexColor("#2563eb"))
    c.rect(px + 300, leg_y, 15, 10, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(px + 320, leg_y + 1, "Posit16 (posit<16,1>)")

    c.save()
    base_png = png_path[:-4]
    subprocess.run(["pdftoppm", "-png", "-r", "150", pdf_path, base_png], check=True)
    subprocess.run(["mv", f"{base_png}-1.png", png_path], check=True)
    print(f"Generated: {png_path}")

def plot_per_node_error_distribution():
    """Plot per-node L2 error distribution: sorted scatter of FP16 vs Posit16 errors"""
    import csv as csv_mod

    nodes, fp16_errs, pos16_errs = [], [], []
    with open("results/logs/per_node_numerical_error.csv") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            nodes.append(int(row['node']))
            fp16_errs.append(float(row['fp16_l2_error']))
            pos16_errs.append(float(row['posit16_l2_error']))

    # Sort by FP16 error descending for visual clarity
    combined = sorted(zip(fp16_errs, pos16_errs), key=lambda x: -x[0])
    fp16_sorted = [x[0] for x in combined]
    pos16_sorted = [x[1] for x in combined]

    pdf_path = "results/plots/master_summary/per_node_error_distribution.pdf"
    png_path = "results/plots/master_summary/per_node_error_distribution.png"

    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(px, height - 55, "Per-Node L2 Logit Error Distribution (2708 Nodes)")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(px, height - 72, "Sorted by FP16 error (descending). Lower is better.")

    # Background
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.rect(px, py, panel_w, panel_h, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(1)
    c.rect(px, py, panel_w, panel_h, fill=0, stroke=1)

    max_err = max(max(fp16_sorted), max(pos16_sorted)) * 1.1

    # Y gridlines
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#64748b"))
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y_val = max_err * frac
        y_pt = py + frac * panel_h
        c.setStrokeColor(colors.HexColor("#e2e8f0"))
        c.setLineWidth(0.5)
        c.line(px, y_pt, px + panel_w, y_pt)
        c.drawString(px - 40, y_pt - 3, f"{y_val:.1e}")

    # X-axis
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#334155"))
    c.drawString(px + panel_w/2 - 40, py - 30, "Node Index (sorted)")

    c.saveState()
    c.translate(px - 48, py + panel_h/2)
    c.rotate(90)
    c.drawString(-30, 0, "L2 Error")
    c.restoreState()

    # Draw with clipping
    c.saveState()
    clip_p = c.beginPath()
    clip_p.rect(px, py, panel_w, panel_h)
    c.clipPath(clip_p, stroke=0)

    n_nodes = len(fp16_sorted)

    # Draw FP16 dots (Red, small)
    c.setFillColor(colors.HexColor("#ef4444"))
    for i in range(0, n_nodes, 3):  # every 3rd for performance
        x_pt = px + (i / n_nodes) * panel_w
        y_pt = py + (fp16_sorted[i] / max_err) * panel_h
        c.circle(x_pt, y_pt, 0.8, fill=1, stroke=0)

    # Draw Posit16 dots (Blue, small)
    c.setFillColor(colors.HexColor("#2563eb"))
    for i in range(0, n_nodes, 3):
        x_pt = px + (i / n_nodes) * panel_w
        y_pt = py + (pos16_sorted[i] / max_err) * panel_h
        c.circle(x_pt, y_pt, 0.8, fill=1, stroke=0)

    c.restoreState()

    # Legend below graph
    leg_y = py - 55
    c.setFillColor(colors.HexColor("#ef4444"))
    c.circle(px + 130, leg_y + 5, 4, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(px + 140, leg_y + 1, "FP16 L2 Error")

    c.setFillColor(colors.HexColor("#2563eb"))
    c.circle(px + 300, leg_y + 5, 4, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(px + 310, leg_y + 1, "Posit16 L2 Error")

    c.save()
    base_png = png_path[:-4]
    subprocess.run(["pdftoppm", "-png", "-r", "150", pdf_path, base_png], check=True)
    subprocess.run(["mv", f"{base_png}-1.png", png_path], check=True)
    print(f"Generated: {png_path}")

if __name__ == "__main__":
    print("Generating Regime 1 plots...")
    plot_regime(
        "results/plots/regime1_without_optimizer",
        "results/logs/training_log_200ep_fp16_gd.csv",
        "results/logs/training_log_200ep_posit16_gd.csv",
        "Regime 1 (Vanilla GD)", loss_max=2.5
    )

    print("Generating Regime 2 plots...")
    plot_regime(
        "results/plots/regime2_pure16_optimizer",
        "results/logs/training_log_200ep_fp16_adam16.csv",
        "results/logs/training_log_200ep_posit16_adam16.csv",
        "Regime 2 (Pure 16-Bit Adam)", loss_max=25.0
    )

    print("Generating Regime 3 plots...")
    plot_regime(
        "results/plots/regime3_double_optimizer",
        "results/logs/training_log_200ep_fp16_adam_double.csv",
        "results/logs/training_log_200ep_posit16_adam_double.csv",
        "Regime 3 (Double Adam States)", loss_max=2.5
    )

    print("Generating Master Summary plot...")
    plot_master_summary()

    print("Generating Numerical Accuracy plots...")
    plot_numerical_accuracy()
    plot_per_node_error_distribution()

    print("All plots generated successfully!")

