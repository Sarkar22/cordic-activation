#!/usr/bin/env python3
"""
End-to-end study (paper section 5.5): does the fixed-point CORDIC activation,
at a reduced iteration count, still drive a real classifier well?

We train a small MLP (2-8-1, tanh hidden + sigmoid output) on the classic
two-moons dataset in float64, then run INFERENCE with the bit-accurate CORDIC
activation model (Q2.12) swept over the iteration count, and report test accuracy.

Output: results/nn_accuracy_sweep.csv
Everything is numpy-only and deterministic (seeded).
"""
import math, os
import numpy as np

import cordic_model as cm   # the bit-accurate fixed-point reference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG  = np.random.default_rng(2026)


# ---------------------------------------------------------------- data --------
def make_moons(n=800, noise=0.12):
    n_out = n // 2
    n_in  = n - n_out
    t_out = np.linspace(0, math.pi, n_out)
    t_in  = np.linspace(0, math.pi, n_in)
    Xo = np.c_[np.cos(t_out),            np.sin(t_out)]
    Xi = np.c_[1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5]
    X  = np.vstack([Xo, Xi])
    y  = np.r_[np.zeros(n_out), np.ones(n_in)]
    X += RNG.normal(0, noise, X.shape)
    # standardize then scale so MLP pre-activations stay near the CORDIC range
    X = (X - X.mean(0)) / X.std(0)
    X *= 0.6
    idx = RNG.permutation(n)
    return X[idx], y[idx]


# -------------------------------------------------------------- float MLP -----
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def train_mlp(X, y, H=8, epochs=4000, lr=0.15, l2=2e-3):
    # weight decay keeps pre-activations inside the tanh-curved region (and the
    # CORDIC convergence radius ~1.118) instead of saturating tanh as a sign fn.
    n, d = X.shape
    W1 = RNG.normal(0, 0.5, (d, H)); b1 = np.zeros(H)
    W2 = RNG.normal(0, 0.5, (H, 1)); b2 = np.zeros(1)
    yc = y.reshape(-1, 1)
    for _ in range(epochs):
        z1 = X @ W1 + b1;  a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2; p  = sigmoid(z2)
        g2 = (p - yc) / n
        dW2 = a1.T @ g2 + l2*W2;    db2 = g2.sum(0)
        g1  = (g2 @ W2.T) * (1 - a1**2)
        dW1 = X.T @ g1 + l2*W1;     db1 = g1.sum(0)
        W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
    return W1, b1, W2, b2

def forward_float(X, P):
    W1, b1, W2, b2 = P
    a1 = np.tanh(X @ W1 + b1)
    return sigmoid(a1 @ W2 + b2).ravel()


# ----------------------------------------------- fixed-point CORDIC activations
QF = 1 << cm.IO_F                       # Q2.12 scale
def quantize_arg(x, lo, hi):
    """Clamp to the unit's input range and quantize to a Q2.12 integer code."""
    x = np.clip(x, lo, hi)
    code = np.round(x * QF).astype(np.int64)
    return np.clip(code, -(1 << (cm.IO_W-1)), (1 << (cm.IO_W-1)) - 1)

def cordic_tanh_vec(x, n_hyp, n_div):
    # tanh: hyperbolic CORDIC converges for |arg| <= ~1.118; clamp at 1.0
    codes = quantize_arg(x, -1.0, 1.0)
    return np.array([cm.eval_tanh(int(c), n_hyp, n_div) for c in codes.ravel()],
                    dtype=np.float64).reshape(x.shape) / QF

def cordic_sigmoid_vec(x, n_hyp, n_div):
    # sigmoid uses arg = x/2, so |x| <= 2.0 keeps tanh arg in range
    codes = quantize_arg(x, -2.0, 2.0)
    return np.array([cm.eval_sigmoid(int(c), n_hyp, n_div) for c in codes.ravel()],
                    dtype=np.float64).reshape(x.shape) / QF

def forward_cordic(X, P, n_hyp, n_div):
    W1, b1, W2, b2 = P
    a1 = cordic_tanh_vec(X @ W1 + b1, n_hyp, n_div)
    return cordic_sigmoid_vec(a1 @ W2 + b2, n_hyp, n_div).ravel()

# "ideal" fixed-point unit: identical quantization + input clamp + output Q2.12
# rounding, but EXACT tanh/sigmoid (i.e. the CORDIC limit as iterations -> inf).
# Comparing CORDIC against this isolates the iteration error from the (shared)
# range/quantization error.
def _q_out(v):
    return np.clip(np.round(v * QF), -(1 << (cm.IO_W-1)), (1 << (cm.IO_W-1)) - 1) / QF
def ideal_tanh_vec(x):
    return _q_out(np.tanh(quantize_arg(x, -1.0, 1.0) / QF)).reshape(x.shape)
def ideal_sigmoid_vec(x):
    return _q_out(sigmoid(quantize_arg(x, -2.0, 2.0) / QF)).reshape(x.shape)
def forward_ideal(X, P):
    W1, b1, W2, b2 = P
    a1 = ideal_tanh_vec(X @ W1 + b1)
    return ideal_sigmoid_vec(a1 @ W2 + b2).ravel()


# ------------------------------------------------------------------- run -------
def main():
    X, y = make_moons()
    ntr = int(0.7 * len(X))
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    P = train_mlp(Xtr, ytr)
    pf  = forward_float(Xte, P)
    pid = forward_ideal(Xte, P)                       # ideal fixed-point reference
    lab = (yte > 0.5)
    acc_float = float(((pf > 0.5) == lab).mean())
    acc_ideal = float(((pid > 0.5) == lab).mean())

    z1 = Xtr @ P[0] + P[1]
    print(f"hidden pre-activation range: [{z1.min():.2f}, {z1.max():.2f}]  "
          f"({(np.abs(z1) <= 1.118).mean()*100:.1f}% within convergence)")
    print(f"float MLP test acc = {acc_float*100:.2f}% | "
          f"ideal fixed-point acc = {acc_ideal*100:.2f}%\n")
    print("  imax stages  cordic_acc  vs_labels   prob_MAE(vs ideal)  -> iteration error")

    rows = []
    for imax in range(4, cm.IMAX + 1):
        cm.set_config(imax, imax)
        ph = forward_cordic(Xte, P, cm.NH, imax)
        acc_h = float(((ph > 0.5) == lab).mean())
        mae   = float(np.mean(np.abs(ph - pid)))      # isolates iteration error
        rows.append((imax, cm.NH + imax, acc_h, mae))
        print(f"  {imax:>4} {cm.NH+imax:>5}   {acc_h*100:6.2f}%             {mae:.3e}")

    path = os.path.join(ROOT, "results", "nn_accuracy_sweep.csv")
    with open(path, "w") as f:
        f.write(f"# float MLP test acc = {acc_float:.6f}; ideal fixed-point acc = {acc_ideal:.6f}\n")
        f.write("imax,total_stages,cordic_test_acc,prob_mae_vs_ideal\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6e}\n")
    cm.set_config(cm.IMAX, cm.NDIV)     # restore default
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
