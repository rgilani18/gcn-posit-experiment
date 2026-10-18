import csv
import subprocess
import os
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors

width, height = landscape(letter)  # 792 x 612 pt

def read_data(filename):
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

def draw_compiled_dashboard():
    data = read_data("results/logs/numerical_accuracy_3regimes.csv")
    r1_f, r1_p = data[0], data[1]
    r2_f, r2_p = data[2], data[3]
    r3_f, r3_p = data[4], data[5]

    pdf_path = "results/plots/master_summary/master_compiled_dashboard.pdf"
    png_path = "results/plots/master_summary/master_compiled_dashboard.png"

    c = canvas.Canvas(pdf_path, pagesize=landscape(letter))

    # =========================================================================
    # Header Banner
    # =========================================================================
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(40, height - 38, "Comprehensive GCN Study: Performance & Numerical Accuracy (200 Epochs)")
    
    c.setFont("Helvetica", 9.5)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(40, height - 52, "Cora Citation Network | 2-Layer GCN (1433 -> 16 -> 7) | Seed 2026 | Pure sw::universal vs 80-Bit Long Double Ground Truth")

    # Layout Parameters for 2x2 Grid
    col_w = 345
    row_h = 210
    left_x1 = 45
    left_x2 = 405
    top_y1 = height - 280  # Top row
    top_y2 = 65            # Bottom row

    regime_labels = ["Regime 1\n(Vanilla GD)", "Regime 2\n(Pure 16b Adam)", "Regime 3\n(Double Adam)"]

    def draw_panel_box(px, py, title, y_label):
        c.setFillColor(colors.HexColor("#ffffff"))
        c.rect(px, py, col_w, row_h, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.setLineWidth(1)
        c.rect(px, py, col_w, row_h, fill=0, stroke=1)

        # Panel Title Banner
        c.setFillColor(colors.HexColor("#f1f5f9"))
        c.rect(px, py + row_h - 26, col_w, 26, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.line(px, py + row_h - 26, px + col_w, py + row_h - 26)

        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(colors.HexColor("#1e293b"))
        c.drawString(px + 12, py + row_h - 18, title)

        # Y Axis label
        c.saveState()
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.HexColor("#64748b"))
        c.translate(px - 28, py + row_h/2)
        c.rotate(90)
        c.drawCentredString(0, 0, y_label)
        c.restoreState()

    # =========================================================================
    # Panel 1: Test Accuracy (%) [Top-Left]
    # =========================================================================
    p1_x, p1_y = left_x1, top_y1
    draw_panel_box(p1_x, p1_y, "1. Classification Test Accuracy (%)", "Test Accuracy (%)")

    # Gridlines for Acc (0 to 100%)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#94a3b8"))
    for val in [0, 20, 40, 60, 80, 100]:
        y_pt = p1_y + 25 + (val / 100.0) * 145
        c.setStrokeColor(colors.HexColor("#f1f5f9"))
        c.setLineWidth(0.8)
        c.line(p1_x, y_pt, p1_x + col_w, y_pt)
        c.drawString(p1_x + 5, y_pt - 2, f"{val}%")

    # Bars for Panel 1
    p1_pairs = [(r1_f['test_acc'], r1_p['test_acc']),
                (r2_f['test_acc'], r2_p['test_acc']),
                (r3_f['test_acc'], r3_p['test_acc'])]
    
    group_spacing = col_w / 3.0
    bar_w = 34

    for i, (f_acc, p_acc) in enumerate(p1_pairs):
        gx = p1_x + i * group_spacing + group_spacing / 2.0
        
        # FP16 Bar
        f_h = (f_acc / 100.0) * 145
        c.setFillColor(colors.HexColor("#ef4444"))
        c.rect(gx - bar_w - 3, p1_y + 25, bar_w, f_h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 7.5)
        c.drawCentredString(gx - bar_w/2 - 3, p1_y + 25 + f_h + 3, f"{f_acc:.1f}%")

        # Posit16 Bar
        p_h = (p_acc / 100.0) * 145
        c.setFillColor(colors.HexColor("#2563eb"))
        c.rect(gx + 3, p1_y + 25, bar_w, p_h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.drawCentredString(gx + bar_w/2 + 3, p1_y + 25 + p_h + 3, f"{p_acc:.1f}%")

        # Regime labels below
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.HexColor("#334155"))
        lines = regime_labels[i].split('\n')
        for li, line in enumerate(lines):
            c.drawCentredString(gx, p1_y + 13 - li * 9, line)

    # =========================================================================
    # Panel 2: Validation Loss [Top-Right]
    # =========================================================================
    p2_x, p2_y = left_x2, top_y1
    draw_panel_box(p2_x, p2_y, "2. Cross-Entropy Validation Loss", "Validation Loss")

    # Gridlines (0.0 to 8.0)
    max_loss_scale = 8.0
    for val in [0.0, 2.0, 4.0, 6.0, 8.0]:
        y_pt = p2_y + 25 + (val / max_loss_scale) * 145
        c.setStrokeColor(colors.HexColor("#f1f5f9"))
        c.setLineWidth(0.8)
        c.line(p2_x, y_pt, p2_x + col_w, y_pt)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#94a3b8"))
        c.drawString(p2_x + 5, y_pt - 2, f"{val:.1f}")

    p2_pairs = [(r1_f['val_loss'], r1_p['val_loss']),
                (r2_f['val_loss'], r2_p['val_loss']),
                (r3_f['val_loss'], r3_p['val_loss'])]

    for i, (f_loss, p_loss) in enumerate(p2_pairs):
        gx = p2_x + i * group_spacing + group_spacing / 2.0
        
        # FP16 Bar
        f_h = min((f_loss / max_loss_scale) * 145, 145)
        c.setFillColor(colors.HexColor("#ef4444"))
        c.rect(gx - bar_w - 3, p2_y + 25, bar_w, f_h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 7.5)
        c.drawCentredString(gx - bar_w/2 - 3, p2_y + 25 + f_h + 3, f"{f_loss:.2f}")

        # Posit16 Bar
        p_h = (p_loss / max_loss_scale) * 145
        c.setFillColor(colors.HexColor("#2563eb"))
        c.rect(gx + 3, p2_y + 25, bar_w, p_h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.drawCentredString(gx + bar_w/2 + 3, p2_y + 25 + p_h + 3, f"{p_loss:.2f}")

        # Regime labels
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.HexColor("#334155"))
        lines = regime_labels[i].split('\n')
        for li, line in enumerate(lines):
            c.drawCentredString(gx, p2_y + 13 - li * 9, line)

    # =========================================================================
    # Panel 3: Mean Absolute Error (MAE) vs Long Double [Bottom-Left]
    # =========================================================================
    p3_x, p3_y = left_x1, top_y2
    draw_panel_box(p3_x, p3_y, "3. Logit Mean Absolute Error (MAE vs Long Double)", "MAE (Scientific)")

    p3_pairs = [(r1_f['mae'], r1_p['mae']),
                (r2_f['mae'], r2_p['mae']),
                (r3_f['mae'], r3_p['mae'])]

    for i, (f_mae, p_mae) in enumerate(p3_pairs):
        gx = p3_x + i * group_spacing + group_spacing / 2.0
        max_v = max(f_mae, p_mae)
        scale = 135 / (max_v * 1.35)

        f_h = max(f_mae * scale, 3.0)
        p_h = max(p_mae * scale, 3.0)

        # FP16 Bar
        c.setFillColor(colors.HexColor("#ef4444"))
        c.rect(gx - bar_w - 3, p3_y + 30, bar_w, f_h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 7.0)
        c.drawCentredString(gx - bar_w/2 - 3, p3_y + 30 + f_h + 3, f"{f_mae:.2e}")

        # Posit16 Bar
        c.setFillColor(colors.HexColor("#2563eb"))
        c.rect(gx + 3, p3_y + 30, bar_w, p_h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.drawCentredString(gx + bar_w/2 + 3, p3_y + 30 + p_h + 3, f"{p_mae:.2e}")

        # Ratio Callout
        ratio = f_mae / p_mae
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.HexColor("#059669"))
        c.drawCentredString(gx, p3_y + 30 + max(f_h, p_h) + 14, f"Posit {ratio:.0f}× lower")

        # Labels
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.HexColor("#334155"))
        lines = regime_labels[i].split('\n')
        for li, line in enumerate(lines):
            c.drawCentredString(gx, p3_y + 16 - li * 9, line)

    # =========================================================================
    # Panel 4: Maximum Absolute Error & Disagreements [Bottom-Right]
    # =========================================================================
    p4_x, p4_y = left_x2, top_y2
    draw_panel_box(p4_x, p4_y, "4. Max Absolute Error & Prediction Disagreements", "Max Error (vs Long Double)")

    p4_pairs = [(r1_f['max_err'], r1_p['max_err'], r1_f['disagreements'], r1_p['disagreements']),
                (r2_f['max_err'], r2_p['max_err'], r2_f['disagreements'], r2_p['disagreements']),
                (r3_f['max_err'], r3_p['max_err'], r3_f['disagreements'], r3_p['disagreements'])]

    for i, (f_max, p_max, f_dis, p_dis) in enumerate(p4_pairs):
        gx = p4_x + i * group_spacing + group_spacing / 2.0
        max_v = max(f_max, p_max)
        scale = 135 / (max_v * 1.35)

        f_h = max(f_max * scale, 3.0)
        p_h = max(p_max * scale, 3.0)

        # FP16 Bar
        c.setFillColor(colors.HexColor("#ef4444"))
        c.rect(gx - bar_w - 3, p4_y + 30, bar_w, f_h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 7.0)
        c.drawCentredString(gx - bar_w/2 - 3, p4_y + 30 + f_h + 3, f"{f_max:.2e}")

        # Posit16 Bar
        c.setFillColor(colors.HexColor("#2563eb"))
        c.rect(gx + 3, p4_y + 30, bar_w, p_h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#0f172a"))
        c.drawCentredString(gx + bar_w/2 + 3, p4_y + 30 + p_h + 3, f"{p_max:.2e}")

        # Disagreements Callout Badge
        c.setFont("Helvetica-Bold", 7.5)
        c.setFillColor(colors.HexColor("#dc2626"))
        c.drawCentredString(gx, p4_y + 30 + max(f_h, p_h) + 14, f"Flipped: FP={f_dis} vs Pos={p_dis}")

        # Labels
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.HexColor("#334155"))
        lines = regime_labels[i].split('\n')
        for li, line in enumerate(lines):
            c.drawCentredString(gx, p4_y + 16 - li * 9, line)

    # =========================================================================
    # Master Legend (Bottom Center) - 100% Outside All Graphs
    # =========================================================================
    leg_y = 20
    leg_center_x = width / 2.0

    c.setFillColor(colors.HexColor("#ffffff"))
    c.rect(leg_center_x - 170, leg_y - 8, 340, 26, fill=1, stroke=1)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))

    # FP16 Item
    c.setFillColor(colors.HexColor("#ef4444"))
    c.rect(leg_center_x - 150, leg_y + 1, 14, 10, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(leg_center_x - 130, leg_y + 2, "IEEE FP16 (cfloat<16,5>)")

    # Posit16 Item
    c.setFillColor(colors.HexColor("#2563eb"))
    c.rect(leg_center_x + 30, leg_y + 1, 14, 10, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0f172a"))
    c.drawString(leg_center_x + 50, leg_y + 2, "Posit16 (posit<16,1>)")

    c.save()
    base_png = png_path[:-4]
    subprocess.run(["pdftoppm", "-png", "-r", "150", pdf_path, base_png], check=True)
    subprocess.run(["mv", f"{base_png}-1.png", png_path], check=True)
    print(f"Generated Compiled Dashboard: {png_path}")

if __name__ == "__main__":
    draw_compiled_dashboard()

