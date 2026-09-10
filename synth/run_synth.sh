#!/usr/bin/env bash
# Sweep the CORDIC iteration count and synth+impl each config in Vivado.
# Produces results/synth_sweep.csv  (one row per config) for the paper's
# area / Fmax / power vs accuracy table.
#
# Usage:
#   synth/run_synth.sh                  # full sweep, default Zynq-7000 part
#   PART=xck26-sfvc784-2LV-c synth/run_synth.sh   # KV260 (UltraScale+) for DSP-baseline parity
#   PERIOD=3.0 SYNTH_ONLY=1 synth/run_synth.sh    # fast estimate, no place/route
set -e
cd "$(dirname "$0")/.."

PART="${PART:-xc7z020clg400-1}"     # PYNQ-Z1 / Zynq-7000 (widely installed)
PERIOD="${PERIOD:-2.0}"             # over-constrain to extract Fmax via WNS
SYNTH_ONLY="${SYNTH_ONLY:-0}"

# (imax, ndiv) configs spanning the accuracy/area trade-off.
CONFIGS=( "8 8" "10 10" "12 12" "14 14" "16 16" )

rm -f results/synth_sweep.csv
for cfg in "${CONFIGS[@]}"; do
    read imax ndiv <<< "$cfg"
    label="cordic_i${imax}_d${ndiv}"
    echo "######## $label  (part=$PART period=${PERIOD}ns) ########"
    python3 model/cordic_model.py --imax "$imax" --ndiv "$ndiv" --no-vectors --no-sweep
    vivado -mode batch -nojournal -log "results/synth/${label}.log" \
           -source synth/synth_one.tcl -tclargs "$PART" "$PERIOD" "$label" "$SYNTH_ONLY"
done

# restore the default (highest-accuracy) config + vectors for simulation
python3 model/cordic_model.py >/dev/null
echo "==== sweep done -> results/synth_sweep.csv ===="
column -t -s, results/synth_sweep.csv
