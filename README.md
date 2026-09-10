# Zero-DSP CORDIC Activation Unit (tanh and sigmoid)

One multiplier-free CORDIC datapath that computes **both** `tanh` and `sigmoid` in Q2.12 fixed point, using zero DSP slices and zero block RAM. Includes the exhaustive oracle study of early-exit headroom for the same unit.

This is the artifact for a paper submitted to **IEEE CCECE 2027** (under review). Every
number below is reproduced by the scripts in this repository.

## Results

- **16384/16384 bit-exact** against the golden model, for tanh *and* sigmoid (all input codes)
- Max error ≈ 1.2 LSB of Q2.12; latency 37 cycles
- KV260 (xck26) place-and-route: 1616 LUT / 1814 FF / **0 DSP / 0 BRAM**
- sky130 ASIC: 0.404 mm², 14,891 cells, DRC clean and LVS clean
- Oracle study: over all 2^14 inputs, a *perfect* early-exit oracle saves only 18-23% beyond a saturation-comparator bypass, so early-exit hardware was not built

## Reproducing

```bash
bash sim/run_sim.sh   # exhaustive bit-exactness proof
```
```bash
python3 model/cordic_model.py   # accuracy sweep CSV
```
```bash
python3 model/nn_experiment.py   # end-to-end MLP accuracy
```
```bash
python3 model/oracle_study.py   # exhaustive early-exit oracle bound
```
```bash
PART=xck26-sfvc784-2LV-c bash synth/run_synth.sh   # area / Fmax / power sweep
```

```bash
OPENLANE_DIR=/path/to/OpenLane DESIGN=cordic_activation bash asic/run_asic.sh   # sky130 RTL-to-GDSII
```

The `asic/` directory holds the OpenLane configuration for each
signed-off configuration; the flow itself is third-party (see Requirements).

## Requirements

Third-party tools, not included here. Any recent version should work; these are what the
reported numbers were produced with.

| tool | used | purpose |
|---|---|---|
| Icarus Verilog | 12.0 | simulation (`iverilog -g2012`) |
| Python 3 | 3.12 + numpy | golden models and analysis |
| AMD Vivado | 2024.2 | FPGA synthesis and place-and-route |
| OpenLane | v1.0.2, sky130A PDK | open-source RTL-to-GDSII |

## Layout

```
model/    Python golden model: the executable specification, and the analysis scripts
rtl/      synthesizable SystemVerilog
tb/       self-checking testbenches
sim/      one-command verification
synth/    Vivado scripts (constraints, sweeps, reporting)
results/  measured CSVs and the figures generated from them
```

## Notes

`rtl/cordic_consts.svh` is **generated** by `model/cordic_model.py`; do not hand-edit it.

The 12-DSP polynomial `tanh` baseline used for comparison in the paper is the author's separate [fpga-tanh-activation](https://github.com/Sarkar22/fpga-tanh-activation) project and is not duplicated here.

## License

MIT, see [LICENSE](LICENSE).
