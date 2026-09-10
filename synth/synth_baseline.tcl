# synth_baseline.tcl -- synth+impl the DSP48 polynomial tanh baseline for Table III.
#   vivado -mode batch -source synth/synth_baseline.tcl -tclargs <part> <period_ns>
# Needs an UltraScale+ part (the baseline instantiates DSP48E2 primitives).
set part   [lindex $argv 0]
set period [lindex $argv 1]
set label  "dsp_poly_tanh"
set root   [pwd]
set outdir $root/results/synth/$label
file mkdir $outdir

read_verilog -sv $root/baseline/tanh.sv
synth_design -top tanh -part $part -flatten_hierarchy rebuilt
create_clock -period $period -name clk [get_ports clk]
opt_design
place_design
route_design

report_utilization    -file $outdir/utilization.rpt
report_timing_summary -file $outdir/timing.rpt
report_power          -file $outdir/power.rpt

proc ncells {pat} { return [llength [get_cells -hierarchical -filter "REF_NAME =~ $pat"]] }
set luts [ncells LUT*]; set ff [ncells FD*]; set dsp [ncells DSP48*]
set bram [ncells RAMB*]; set srl [ncells SRL*]
set wns [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
if {$wns eq ""} { set wns 0.0 }
set fmax [expr {($period-$wns) > 0 ? 1000.0/($period-$wns) : 0.0}]
set power "NA"
if {[catch {
    set fh [open $outdir/power.rpt r]; set txt [read $fh]; close $fh
    if {[regexp {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} $txt -> p]} { set power $p }
}]} {}

set csv $root/results/baseline_synth.csv
set fh [open $csv w]
puts $fh "label,part,period_ns,lut,ff,dsp,bram,srl,wns_ns,fmax_mhz,power_w"
puts $fh [format "%s,%s,%s,%d,%d,%d,%d,%d,%.3f,%.1f,%s" \
          $label $part $period $luts $ff $dsp $bram $srl $wns $fmax $power]
close $fh
puts "==== $label : LUT=$luts FF=$ff DSP=$dsp BRAM=$bram SRL=$srl WNS=$wns Fmax=${fmax}MHz Power=${power}W"
