#!/usr/bin/env bash
# Compile + run the exhaustive bit-exact verification (iverilog).
set -e
cd "$(dirname "$0")/.."

echo ">> regenerating golden model (consts + vectors)"
python3 model/cordic_model.py >/dev/null

echo ">> compiling"
iverilog -g2012 -Wall -I rtl -o sim/sim.vvp rtl/cordic_activation.sv tb/tb_cordic.sv

echo ">> tanh"
vvp sim/sim.vvp +vec=sim/vec_tanh.hex +mode=0

echo ">> sigmoid"
vvp sim/sim.vvp +vec=sim/vec_sigmoid.hex +mode=1
