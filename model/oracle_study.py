#!/usr/bin/env python3
"""
Exhaustive ORACLE early-exit study for the CORDIC tanh/sigmoid unit.

Question: if we had a perfect crystal ball telling us the earliest iteration at
which the final Q2.12 output is already decided ("pinned"), how many cycles would
a folded (iterative) engine save -- beyond what a trivial constant-output bypass
already gets -- on (a) all inputs, (b) the non-trivial region, (c) the realistic
MLP workload?

This is pure Python against the bit-exact golden model (cordic-activation).
Viability rule, fixed before the study: if oracle mean savings beat the bypass baseline by
< 30% on the workload distribution, kill the idea before writing any RTL.

Semantics of truncation == hardware early exit:
  eval(x, k, d) runs k hyperbolic stages then d division iterations, exactly what
  a folded engine that retires early would compute.
  Pin stage k*(x) = smallest k s.t. eval(x, j, ND) == eval(x, NH, ND) for ALL j >= k
  (suffix-stable, so a sound detector stopping at k* is always correct).
"""
import os, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cordic_model as cm
import nn_experiment as nx
import numpy as np

NH, ND = cm.NH, cm.NDIV            # 18 hyperbolic stages (with repeats), 16 div iters
OVH = 3                            # in/out + handoff registers (latency = 3+NH+ND = 37)
FULL_CYC = OVH + NH + ND
LO, HI = -(1 << (cm.IO_W - 1)), (1 << (cm.IO_W - 1)) - 1   # 14-bit code range
QF = 1 << cm.IO_F

def study(fn, name):
    codes = range(LO, HI + 1)
    full = {}
    khyp, ddiv, joint_ok, cyc = {}, {}, {}, {}
    for x in codes:
        f = fn(x, NH, ND)
        full[x] = f
        # earliest suffix-stable hyperbolic stage (division run in full)
        outs = [fn(x, k, ND) for k in range(NH + 1)]
        k = NH
        while k > 0 and outs[k - 1] == f:
            k -= 1
        # ...suffix check: ensure all j >= k match (outs may oscillate)
        while any(outs[j] != f for j in range(k, NH + 1)):
            k += 1
        khyp[x] = k
        # earliest suffix-stable division iteration (hyperbolic run in full)
        outs_d = [fn(x, NH, d) for d in range(ND + 1)]
        d = ND
        while d > 0 and outs_d[d - 1] == f:
            d -= 1
        while any(outs_d[j] != f for j in range(d, ND + 1)):
            d += 1
        ddiv[x] = d
        # joint point-check: stopping BOTH at (k, d) still exact?
        ok = fn(x, k, d) == f
        joint_ok[x] = ok
        cyc[x] = OVH + (k + d if ok else min(k + ND, NH + d))
    # trivial constant tails (what a 1-cycle comparator+constant bypass may claim)
    hi_t = HI
    while hi_t > 0 and full[hi_t - 1] == full[HI]:
        hi_t -= 1
    lo_t = LO
    while lo_t < 0 and full[lo_t + 1] == full[LO]:
        lo_t += 1
    in_bypass = lambda x: (x >= hi_t) or (x <= lo_t)
    n_bypass = sum(1 for x in codes if in_bypass(x))
    print(f"\n=== {name} ===")
    print(f"constant tails: x <= {lo_t} ({lo_t/QF:+.4f}) out={full[LO]}, "
          f"x >= {hi_t} ({hi_t/QF:+.4f}) out={full[HI]}  "
          f"-> bypass covers {n_bypass}/{HI-LO+1} codes ({100*n_bypass/(HI-LO+1):.1f}%)")
    print(f"joint (k*,d*) exact for {sum(joint_ok.values())}/{HI-LO+1} codes")
    return dict(full=full, khyp=khyp, ddiv=ddiv, cyc=cyc,
                in_bypass=in_bypass, lo_t=lo_t, hi_t=hi_t)

def dist_stats(name, R, weights):
    """weights: dict code -> count (a distribution over codes)"""
    tot = sum(weights.values())
    mean_oracle = sum(R["cyc"][x] * w for x, w in weights.items()) / tot
    mean_bypass = sum((1 if R["in_bypass"](x) else FULL_CYC) * w
                      for x, w in weights.items()) / tot
    # oracle engine ALSO gets the bypass for free (comparator): min of both
    mean_both = sum((1 if R["in_bypass"](x) else R["cyc"][x]) * w
                    for x, w in weights.items()) / tot
    extra = 100 * (mean_bypass - mean_both) / mean_bypass
    print(f"  {name:34s} oracle {mean_oracle:5.1f} | bypass-only {mean_bypass:5.1f} | "
          f"oracle+bypass {mean_both:5.1f} cyc "
          f"({100*(1-mean_both/FULL_CYC):4.1f}% vs full, +{extra:4.1f}% beyond bypass)")
    return mean_both, mean_bypass, extra

def uniform(pred=lambda x: True):
    return {x: 1 for x in range(LO, HI + 1) if pred(x)}

def main():
    # ---- realistic workload: two-moons MLP pre-activations (test set) --------
    X, y = nx.make_moons()
    ntr = int(0.7 * len(X))
    P = nx.train_mlp(X[:ntr], y[:ntr])
    W1, b1, W2, b2 = P
    Xte = X[ntr:]
    z1 = Xte @ W1 + b1                                   # 240 x 8 tanh args
    t_codes = np.clip(np.round(np.clip(z1, -1.0, 1.0) * QF), LO, HI).astype(int)
    a1 = np.array([[cm.eval_tanh(int(c)) for c in row] for row in t_codes]) / QF
    z2 = (a1 @ W2 + b2).ravel()                          # 240 sigmoid args
    s_codes = np.clip(np.round(np.clip(z2, -2.0, 2.0) * QF), LO, HI).astype(int)
    wl_tanh, wl_sig = {}, {}
    for c in t_codes.ravel():
        wl_tanh[int(c)] = wl_tanh.get(int(c), 0) + 1
    for c in s_codes:
        wl_sig[int(c)] = wl_sig.get(int(c), 0) + 1

    results = {}
    for fn, name, wl in ((cm.eval_tanh, "tanh", wl_tanh),
                         (cm.eval_sigmoid, "sigmoid", wl_sig)):
        R = study(fn, name)
        results[name] = R
        print(f"  full latency = {FULL_CYC} cyc; distributions:")
        dist_stats("uniform (all 16384 codes)", R, uniform())
        dist_stats("non-trivial region only", R,
                   uniform(lambda x, R=R: not R["in_bypass"](x)))
        dist_stats(f"MLP workload ({sum(wl.values())} evals)", R, wl)
        ks = np.array([R["khyp"][x] for x in range(LO, HI + 1)])
        ds = np.array([R["ddiv"][x] for x in range(LO, HI + 1)])
        print(f"  k* (hyp stages needed): mean {ks.mean():.1f} / max {ks.max()} of {NH}")
        print(f"  d* (div iters  needed): mean {ds.mean():.1f} / max {ds.max()} of {ND}")
        # dump per-code CSV for later figures
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        with open(os.path.join(ROOT, "results", f"oracle_{name}.csv"), "w") as f:
            w = csv.writer(f)
            w.writerow(["code", "x", "out", "khyp", "ddiv", "cycles_oracle", "bypass"])
            for x in range(LO, HI + 1):
                w.writerow([x, f"{x/QF:.6f}", R["full"][x], R["khyp"][x],
                            R["ddiv"][x], R["cyc"][x], int(R["in_bypass"](x))])
    print("\nwrote results/oracle_tanh.csv, results/oracle_sigmoid.csv")

if __name__ == "__main__":
    main()
