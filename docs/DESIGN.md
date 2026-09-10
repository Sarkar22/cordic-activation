# Design notes — CORDIC activation unit

## 1. Number formats
- **I/O:** Q2.12 signed, 14-bit (range [-2, 2), LSB = 2^-12 ≈ 2.44e-4). Matches the
  DSP polynomial tanh baseline so comparisons are apples-to-apples.
- **Internal:** Q(WI-1-QI).QI signed, `WI=22`, `QI=16` → 4 guard fraction bits over
  the I/O and ample integer headroom (intermediate magnitudes stay < 2).

## 2. Algorithm
### Stage A — hyperbolic CORDIC rotation (compute sinh, cosh)
Seed `x0 = 1.0`, `y0 = 0`, `z0 = arg`. For each scheduled shift `i`:
```
d   = sign(z)                       # +1 if z>=0 else -1
x  += d · (y >> i)
y  += d · (x_old >> i)
z  -= d · atanh(2^-i)
```
After `N_hyp` iterations: `x → K·cosh(arg)`, `y → K·sinh(arg)`, where
`K = Π sqrt(1 - 2^-2i) ≈ 0.8281`.

**Convergence:** hyperbolic CORDIC converges only for `|arg| ≤ Σ atanh(2^-i) ≈ 1.1182`,
and requires **repeated iterations** at `i = 4, 13, 40, …` (`i_{k+1}=3i_k+1`) to stay
convergent. The schedule (with repeats) is `[1,2,3,4,4,5,…,13,13,14,15,16]` for IMAX=16.

### Stage B — linear CORDIC vectoring (divide y/x = tanh)
Seed `xq=x` (constant), `yq=y`, `zq=0`. For `i = 1..N_div`:
```
d   = sign(yq)
yq -= d · (xq >> i)
zq += d · 2^-i
```
`zq → yq/xq = sinh/cosh = tanh(arg)`. **The K scale cancels in the ratio**, so we do
not need to seed x0 with 1/K — `x0 = 1.0` is fine, which also makes the accuracy
sweep clean (fewer iterations only lowers precision, no scale bookkeeping).

### Sigmoid reuse
`σ(x) = ½(1 + tanh(x/2))`. Implemented as: pre-shift the input by one extra bit
(`x/2`), run the identical tanh datapath, then `σ = (t >> 1) + 0.5` at the output.
No extra multipliers, just an input/output mux selected by `i_mode`.
Because the tanh arg is `x/2`, sigmoid stays convergent for `|x| ≤ 2.236`.

## 3. Microarchitecture
- Fully pipelined: input reg → `N_hyp` hyperbolic stages → seed reg → `N_div`
  division stages → output reg. **1 sample/cycle**, latency `1+N_hyp+1+N_div+1`.
- Zero DSP, zero BRAM: every operation is an add/sub or a hard-wired shift
  (constant shift amounts ⇒ pure routing, no barrel shifters).
- Backpressure: a single global enable = `i_ready` freezes the whole pipeline;
  `o_ready = i_ready`. Valid (and the mode bit) shift down with the data.
- Output rounding: round-half-up from QI to Q2.12, then saturate to 14-bit signed.

## 4. Why this is a good trade study
- **DSP-free:** small/IoT FPGAs are DSP-limited; the baseline burns 12 DSPs. We
  quantify the LUT/FF cost of using zero.
- **Tunable:** one knob (iterations) sweeps a clean accuracy↔area Pareto.
- **Graceful degradation:** outside its range, CORDIC saturates (tracking the true
  tanh trend), whereas the odd-polynomial baseline diverges — a real qualitative win.

## 5. Folded (area-optimized) variant — future
The pipelined core is the throughput-optimal point. A folded/iterative version
(one adder set, reused over `N_hyp+N_div` cycles, 1/(N) throughput) gives the
low-area Pareto endpoint; it shares the same constants/golden model. Listed as
future work in the paper; straightforward to add from this RTL.

## 6. Verification
`model/cordic_model.py` is the executable spec. It emits `rtl/cordic_consts.svh`
(so the RTL uses identical constants) and `sim/vec_*.hex` (input,expected for all
2^14 codes). `tb/tb_cordic.sv` drives every input and asserts exact equality.
Result: **0 mismatches** for both tanh and sigmoid.
