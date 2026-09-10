#!/usr/bin/env python3
"""
Generate the paper figures from the result CSVs. Headless (Agg) -> results/figures/*.png + *.pdf.

Run with a matplotlib-enabled interpreter, e.g.:
    /tmp/mplvenv/bin/python model/plot_results.py
(matplotlib is blocked from the system Python on this machine by PEP 668; a
throwaway venv in /tmp works because the data disk is mounted noexec.)
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
LSB  = 2.0 ** -12

def load(name):
    path = os.path.join(ROOT, "results", name)
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.rstrip("\n"))
    rd = csv.DictReader(rows)
    return [ {k: v for k, v in r.items()} for r in rd ]

def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote results/figures/{stem}.png/.pdf")

# --- Fig 2: accuracy vs iterations -------------------------------------------
acc = load("accuracy_sweep.csv")
if acc:
    x  = [int(r["total_stages"]) for r in acc]
    tt = [float(r["tanh_maxabs"]) for r in acc]
    ss = [float(r["sigmoid_maxabs"]) for r in acc]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.semilogy(x, tt, "o-", label="tanh  max |err|")
    ax.semilogy(x, ss, "s-", label="sigmoid  max |err|")
    ax.axhline(LSB, ls="--", color="gray", lw=1)
    ax.text(x[0], LSB*1.15, "1 LSB (Q2.12)", color="gray", fontsize=8)
    ax.set_xlabel("total CORDIC stages"); ax.set_ylabel("max abs error")
    ax.grid(True, which="both", ls=":", lw=0.5); ax.legend(fontsize=8)
    save(fig, "fig_accuracy")

# --- Fig 3: area + Fmax vs iterations ----------------------------------------
def imax_of(label):  # cordic_i12_d12 -> 12
    return int(label.split("_")[1][1:])
syn = load("synth_sweep.csv")
if syn:
    syn = sorted(syn, key=lambda r: imax_of(r["label"]))
    im   = [imax_of(r["label"]) for r in syn]
    lut  = [int(r["lut"]) for r in syn]
    ff   = [int(r["ff"])  for r in syn]
    fmax = [float(r["fmax_mhz"]) for r in syn]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(im, lut, "o-", label="LUT")
    ax.plot(im, ff,  "s-", label="FF")
    ax.set_xlabel("max hyperbolic index (imax)"); ax.set_ylabel("primitives")
    ax.grid(True, ls=":", lw=0.5)
    ax2 = ax.twinx(); ax2.plot(im, fmax, "^--", color="green", label="Fmax")
    ax2.set_ylabel("Fmax (MHz)", color="green")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=8, loc="upper left")
    save(fig, "fig_area")

    # --- Fig 4: accuracy-area Pareto -----------------------------------------
    if acc:
        amap = {int(r["imax"]): float(r["tanh_maxabs"]) for r in acc}
        px = [int(r["lut"]) for r in syn if imax_of(r["label"]) in amap]
        py = [amap[imax_of(r["label"])] for r in syn if imax_of(r["label"]) in amap]
        pl = [imax_of(r["label"]) for r in syn if imax_of(r["label"]) in amap]
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.semilogy(px, py, "o-")
        for xx, yy, ll in zip(px, py, pl):
            ax.annotate(f"imax={ll}", (xx, yy), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("LUTs"); ax.set_ylabel("tanh max abs error")
        ax.grid(True, which="both", ls=":", lw=0.5)
        save(fig, "fig_pareto")

# --- Fig 5: end-to-end NN -----------------------------------------------------
nn = load("nn_accuracy_sweep.csv")
if nn:
    x   = [int(r["total_stages"]) for r in nn]
    mae = [float(r["prob_mae_vs_ideal"]) for r in nn]
    acc2= [float(r["cordic_test_acc"])*100 for r in nn]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.semilogy(x, mae, "o-", color="C3", label="prob MAE vs ideal")
    ax.set_xlabel("total CORDIC stages"); ax.set_ylabel("prob MAE vs ideal FxP", color="C3")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax2 = ax.twinx(); ax2.plot(x, acc2, "s--", color="C0", label="test accuracy")
    ax2.set_ylabel("test accuracy (%)", color="C0")
    save(fig, "fig_nn")

print("done.")
