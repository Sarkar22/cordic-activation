#!/usr/bin/env python3
"""
Bit-accurate fixed-point reference model for the CORDIC activation unit.

This is the SINGLE SOURCE OF TRUTH. It:
  1. Derives the hyperbolic-CORDIC iteration schedule + atanh constants.
  2. Emits  rtl/cordic_consts.svh   (consumed by the RTL, so RTL == model).
  3. Emits  sim/vec_tanh.hex / sim/vec_sigmoid.hex  (exhaustive test vectors).
  4. Emits  results/accuracy_sweep.csv  (accuracy vs #iterations, vs math.*).

Algorithm (both activations share one datapath, ZERO multipliers):
  tanh(z) = sinh(z) / cosh(z)
    Stage A  - hyperbolic CORDIC rotation (shift-add): (cosh*K, sinh*K)
    Stage B  - linear  CORDIC vectoring  (shift-add): divides y/x -> tanh
               (the K scale factor cancels in the ratio, so we need no 1/K seed)
  sigmoid(z) = 0.5*(1 + tanh(z/2))   -> input pre-shift, output shift+bias

Number formats:
  I/O      : Q2.12  signed, 14 bit   (matches the existing DSP tanh baseline)
  internal : Q(WI-1-QI).QI signed, WI bit  (guard bits for accuracy)
"""

import math
import os

# ----------------------------------------------------------------------------
# Configuration  (DUT = highest-accuracy point; sweep configs derived below)
# ----------------------------------------------------------------------------
IO_W   = 14          # I/O word width  (Q2.12)
IO_F   = 12          # I/O fraction bits
WI     = 22          # internal datapath width
QI     = 16          # internal fraction bits  (4 guard bits over IO)
IMAX   = 16          # max hyperbolic shift index for the DUT
NDIV   = 16          # linear-division iterations for the DUT  (<= QI)

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------
# Fixed-point helpers (mirror RTL two's-complement / arithmetic-shift exactly)
# ----------------------------------------------------------------------------
def wrap(v, w):
    """Wrap to w-bit two's complement (RTL register truncation)."""
    v &= (1 << w) - 1
    if v >= (1 << (w - 1)):
        v -= (1 << w)
    return v

def asr(v, s):
    """Arithmetic shift right; Python '>>' floors == RTL '>>>' on signed."""
    return v >> s

def to_fixed(real, frac):
    return int(round(real * (1 << frac)))

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ----------------------------------------------------------------------------
# Build the hyperbolic schedule (with the mandatory repeats at i = 4,13,40,...)
# and the atanh(2^-i) constant table, in internal QI fixed point.
# ----------------------------------------------------------------------------
def build_schedule(imax):
    repeats = set()
    k = 4
    while k <= imax:
        repeats.add(k)
        k = 3 * k + 1
    shifts = []
    for i in range(1, imax + 1):
        shifts.append(i)
        if i in repeats:
            shifts.append(i)            # repeated iteration for convergence
    return shifts

HSHIFT = build_schedule(IMAX)                      # e.g. [1,2,3,4,4,5,...,13,13,14,15,16]
HATANH = [to_fixed(math.atanh(2.0 ** -i), QI) for i in HSHIFT]
NH     = len(HSHIFT)
X0     = to_fixed(1.0, QI)                          # seed x0 = 1.0 (scale cancels)

# Convergence radius = sum of atanh(2^-i) over the (de-duplicated) schedule.
CONV_RADIUS = sum(math.atanh(2.0 ** -i) for i in range(1, IMAX + 1)) \
              + sum(math.atanh(2.0 ** -i) for i in {4, 13} if i <= IMAX)


# ----------------------------------------------------------------------------
# The core CORDIC datapath (one config = (n_hyp, n_div) prefix of full tables)
# ----------------------------------------------------------------------------
def set_config(imax, ndiv):
    """Reconfigure the global schedule/tables (used by the synth sweep)."""
    global IMAX, NDIV, HSHIFT, HATANH, NH, CONV_RADIUS
    IMAX, NDIV = imax, ndiv
    HSHIFT = build_schedule(imax)
    HATANH = [to_fixed(math.atanh(2.0 ** -i), QI) for i in HSHIFT]
    NH = len(HSHIFT)
    CONV_RADIUS = sum(math.atanh(2.0 ** -i) for i in range(1, imax + 1)) \
                  + sum(math.atanh(2.0 ** -i) for i in {4, 13} if i <= imax)

def cordic_tanh_internal(z0, n_hyp=None, n_div=None):
    """z0 is QI fixed point; returns tanh(z0) in QI fixed point (bit-exact)."""
    if n_hyp is None: n_hyp = NH
    if n_div is None: n_div = NDIV
    x, y, z = X0, 0, z0
    for s in range(n_hyp):                          # Stage A: hyperbolic rotation
        sh = HSHIFT[s]
        d  = 1 if z >= 0 else -1
        xn = wrap(x + d * asr(y, sh), WI)
        yn = wrap(y + d * asr(x, sh), WI)
        zn = wrap(z - d * HATANH[s], WI)
        x, y, z = xn, yn, zn
    xq, yq, zq = x, y, 0                             # Stage B: linear division y/x
    for i in range(1, n_div + 1):
        d  = 1 if yq >= 0 else -1
        yq = wrap(yq - d * asr(xq, i), WI)
        zq = wrap(zq + d * (1 << (QI - i)), WI)
    return zq                                        # ~ tanh in QI

def ix_to_internal(i_x, shift):
    """Sign-extend 14-bit Q2.12 input and align to QI, optionally pre-divided."""
    return wrap(i_x << (QI - IO_F - shift), WI)      # shift=0 tanh, shift=1 sigmoid

def internal_to_io(v):
    sh = QI - IO_F                                   # round-half-up, saturate
    r  = (v + (1 << (sh - 1))) >> sh
    return clamp(r, -(1 << (IO_W - 1)), (1 << (IO_W - 1)) - 1)

def eval_tanh(i_x, n_hyp=None, n_div=None):
    return internal_to_io(cordic_tanh_internal(ix_to_internal(i_x, 0), n_hyp, n_div))

def eval_sigmoid(i_x, n_hyp=None, n_div=None):
    t = cordic_tanh_internal(ix_to_internal(i_x, 1), n_hyp, n_div)   # tanh(z/2)
    s = wrap(asr(t, 1) + (1 << (QI - 1)), WI)                         # 0.5*(1+t)
    return internal_to_io(s)


# ----------------------------------------------------------------------------
# Emitters
# ----------------------------------------------------------------------------
def signed_hex(v, w):
    return f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def emit_consts():
    path = os.path.join(ROOT, "rtl", "cordic_consts.svh")
    # element k lives at bits [k*fieldw +: fieldw]; Verilog concat is MSB-first,
    # so emit the elements in reverse order.
    def flat(name, values, fieldw):
        parts = [f"{fieldw}'d{v & ((1 << fieldw) - 1)}" for v in reversed(values)]
        return (f"localparam logic [{len(values)*fieldw-1}:0] {name} = "
                f"{{{', '.join(parts)}}};\n")
    with open(path, "w") as f:
        f.write("// AUTO-GENERATED by model/cordic_model.py -- DO NOT EDIT.\n")
        f.write("// Hyperbolic + linear CORDIC constants (internal QI fixed point).\n")
        f.write("// NOTE: no include guard on purpose -- these are module-scoped\n")
        f.write("// localparams and must be re-included inside every module.\n")
        f.write(f"// schedule (shift per hyperbolic stage): {HSHIFT}\n\n")
        f.write(f"localparam int IO_W = {IO_W};\n")
        f.write(f"localparam int IO_F = {IO_F};\n")
        f.write(f"localparam int WI   = {WI};\n")
        f.write(f"localparam int QI   = {QI};\n")
        f.write(f"localparam int SHW  = 8;  // bits per HSHIFT field\n")
        f.write(f"localparam int NH_MAX = {NH};   // hyperbolic stages (with repeats)\n")
        f.write(f"localparam int ND_MAX = {NDIV}; // linear-division stages\n")
        f.write(f"localparam logic signed [WI-1:0] X0 = {WI}'sd{X0};\n\n")
        f.write("// HSHIFT[k] = HSHIFT_FLAT[k*SHW +: SHW]\n")
        f.write(flat("HSHIFT_FLAT", HSHIFT, 8))
        f.write("\n// HATANH[k] = $signed(HATANH_FLAT[k*WI +: WI])  (all positive)\n")
        f.write(flat("HATANH_FLAT", HATANH, WI))
    return path

def emit_vectors():
    paths = []
    for name, fn in (("tanh", eval_tanh), ("sigmoid", eval_sigmoid)):
        path = os.path.join(ROOT, "sim", f"vec_{name}.hex")
        with open(path, "w") as f:
            for code in range(1 << IO_W):                 # exhaustive: all 2^14 inputs
                i_x = code - (1 << IO_W) if code >= (1 << (IO_W - 1)) else code
                o   = fn(i_x)
                f.write(f"{signed_hex(i_x, IO_W)} {signed_hex(o, IO_W)}\n")
        paths.append(path)
    return paths

def emit_accuracy_sweep():
    """Accuracy of the *model* vs math.tanh/sigmoid over the convergent range,
    swept over iteration count -- this is the paper's accuracy-vs-N table."""
    path = os.path.join(ROOT, "results", "accuracy_sweep.csv")
    lo = -to_fixed(1.0, IO_F)                              # validated range |x| <= 1.0
    hi =  to_fixed(1.0, IO_F)
    rows = []
    for imax in range(4, IMAX + 1):
        sched = build_schedule(imax)
        # temporarily swap global tables to this prefix config
        global HSHIFT, HATANH, NH
        HSHIFT_b, HATANH_b, NH_b = HSHIFT, HATANH, NH
        HSHIFT = sched
        HATANH = [to_fixed(math.atanh(2.0 ** -i), QI) for i in sched]
        NH = len(sched)
        n_div = imax
        et = es = 0.0
        mt = ms = 0.0
        n = 0
        for code in range(lo, hi + 1):
            xr = code / (1 << IO_F)
            t  = eval_tanh(code, NH, n_div)   / (1 << IO_F)
            s  = eval_sigmoid(code, NH, n_div) / (1 << IO_F)
            dt = abs(t - math.tanh(xr))
            ds = abs(s - 1.0 / (1.0 + math.exp(-xr)))
            et += dt * dt; es += ds * ds
            mt = max(mt, dt); ms = max(ms, ds)
            n += 1
        rows.append((imax, NH + n_div, math.sqrt(et / n), mt, math.sqrt(es / n), ms))
        HSHIFT, HATANH, NH = HSHIFT_b, HATANH_b, NH_b
    with open(path, "w") as f:
        f.write("imax,total_stages,tanh_rmse,tanh_maxabs,sigmoid_rmse,sigmoid_maxabs\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6e},{r[3]:.6e},{r[4]:.6e},{r[5]:.6e}\n")
    return path, rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CORDIC activation golden model")
    ap.add_argument("--imax", type=int, default=IMAX, help="max hyperbolic shift index")
    ap.add_argument("--ndiv", type=int, default=NDIV, help="linear-division iterations")
    ap.add_argument("--no-vectors", action="store_true", help="skip test-vector emit")
    ap.add_argument("--no-sweep",   action="store_true", help="skip accuracy sweep")
    args = ap.parse_args()
    set_config(args.imax, args.ndiv)

    c = emit_consts()
    v = emit_vectors() if not args.no_vectors else []
    rows = []
    if not args.no_sweep:
        a, rows = emit_accuracy_sweep()
    print(f"config: WI={WI} QI={QI} IMAX={IMAX} NH={NH} NDIV={NDIV}")
    print(f"convergence radius |z| <= {CONV_RADIUS:.4f}")
    print(f"wrote {os.path.relpath(c, ROOT)}")
    for p in v:
        print(f"wrote {os.path.relpath(p, ROOT)}  ({1 << IO_W} vectors)")
    if rows:
        print(f"wrote {os.path.relpath(a, ROOT)}")
        print("\naccuracy vs math.* over |x|<=1.0 (DUT row = last):")
        print("  imax  stages   tanh_rmse  tanh_max   sig_rmse   sig_max")
        for r in rows:
            print(f"  {r[0]:>4}  {r[1]:>5}   {r[2]:.2e}  {r[3]:.2e}  {r[4]:.2e}  {r[5]:.2e}")
