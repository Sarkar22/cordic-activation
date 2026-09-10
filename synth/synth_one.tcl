# synth_one.tcl -- non-project synth+impl of cordic_activation, emits one CSV row.
#
#   vivado -mode batch -source synth/synth_one.tcl -tclargs <part> <period_ns> <label> [synth_only]
#
# Assumes rtl/cordic_consts.svh has already been (re)generated for this config
# by model/cordic_model.py.  Reports go to results/synth/<label>/, and a summary
# row is appended to results/synth_sweep.csv.

set part      [lindex $argv 0]
set period    [lindex $argv 1]
set label     [lindex $argv 2]
set synthonly [expr {[llength $argv] > 3 ? [lindex $argv 3] : 0}]

set root    [pwd]
set outdir  $root/results/synth/$label
file mkdir  $outdir

read_verilog -sv $root/rtl/cordic_activation.sv
synth_design -top cordic_activation -part $part -include_dirs $root/rtl -flatten_hierarchy rebuilt
create_clock -period $period -name clk [get_ports clk]

if {!$synthonly} {
    opt_design
    place_design
    route_design
}

report_utilization     -file $outdir/utilization.rpt
report_timing_summary  -file $outdir/timing.rpt
report_power           -file $outdir/power.rpt

# ---- scrape metrics ----
proc ncells {pat} { return [llength [get_cells -hierarchical -filter "REF_NAME =~ $pat"]] }
set luts [ncells LUT*]
set ff   [ncells FD*]
set dsp  [ncells DSP48*]
set bram [ncells RAMB*]
set srl  [ncells SRL*]

set wns [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
if {$wns eq ""} { set wns 0.0 }
set achieved [expr {$period - $wns}]
set fmax [expr {$achieved > 0 ? 1000.0/$achieved : 0.0}]

# total on-chip power from the power report
set power "NA"
if {[catch {
    set fh [open $outdir/power.rpt r]; set txt [read $fh]; close $fh
    if {[regexp {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} $txt -> p]} { set power $p }
}]} {}

set csv $root/results/synth_sweep.csv
if {![file exists $csv]} {
    set fh [open $csv w]
    puts $fh "label,part,period_ns,lut,ff,dsp,bram,srl,wns_ns,fmax_mhz,power_w"
    close $fh
}
set fh [open $csv a]
puts $fh [format "%s,%s,%s,%d,%d,%d,%d,%d,%.3f,%.1f,%s" \
          $label $part $period $luts $ff $dsp $bram $srl $wns $fmax $power]
close $fh

puts "==== $label : LUT=$luts FF=$ff DSP=$dsp BRAM=$bram SRL=$srl  WNS=$wns Fmax=${fmax}MHz  Power=${power}W"
