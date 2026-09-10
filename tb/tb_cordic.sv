/*
 * tb_cordic.sv -- exhaustive bit-exact check of cordic_activation against the
 * golden vectors emitted by model/cordic_model.py.
 *
 *   vvp sim.vvp +vec=sim/vec_tanh.hex    +mode=0
 *   vvp sim.vvp +vec=sim/vec_sigmoid.hex +mode=1
 *
 * Each vector line is "<input_hex> <expected_hex>" (Q2.12, 14-bit). The DUT
 * output must EXACTLY equal the expected value for every one of the 2^14 inputs.
 */
`timescale 1ns/1ps
module tb_cordic;
    `include "cordic_consts.svh"
    localparam int MAXN = (1 << IO_W);
    localparam int LAT  = 1 + NH_MAX + 1 + ND_MAX + 1;   // pipeline latency
    localparam     CLK  = 4;

    logic clk = 0, rst, i_valid, i_mode, i_ready;
    logic [13:0] i_x, o_fx;
    logic o_ready, o_valid;

    always #(CLK/2) clk = ~clk;

    cordic_activation dut (
        .clk(clk), .rst(rst), .i_x(i_x), .i_mode(i_mode),
        .i_valid(i_valid), .o_ready(o_ready),
        .o_fx(o_fx), .o_valid(o_valid), .i_ready(i_ready)
    );

    // Load vectors: $readmemh fills tokens sequentially -> tmp[2i]=in, tmp[2i+1]=exp
    logic [13:0] tmp [0:2*MAXN-1];
    logic [13:0] in_x [0:MAXN-1];
    logic [13:0] exp  [0:MAXN-1];
    string vecfile;
    int    mode_arg;

    integer n, oidx, errors, matched;

    initial begin
        if (!$value$plusargs("vec=%s", vecfile)) begin
            $display("FATAL: pass +vec=<file>"); $finish;
        end
        if (!$value$plusargs("mode=%d", mode_arg)) mode_arg = 0;
        $readmemh(vecfile, tmp);
        for (n = 0; n < MAXN; n = n + 1) begin
            in_x[n] = tmp[2*n];
            exp[n]  = tmp[2*n+1];
        end
    end

    // Driver
    initial begin
        rst = 1; i_valid = 0; i_mode = 0; i_x = 0; i_ready = 1;
        repeat (4) @(negedge clk);
        i_mode = mode_arg[0];
        rst = 0;
        for (n = 0; n < MAXN; n = n + 1) begin
            @(negedge clk);
            i_x = in_x[n]; i_valid = 1; i_mode = mode_arg[0];
        end
        @(negedge clk); i_valid = 0;
        repeat (LAT + 16) @(negedge clk);
        $display("FATAL: timed out before receiving all outputs"); $finish;
    end

    // Checker
    initial begin
        errors = 0; oidx = 0; matched = 0;
        @(negedge rst);
        forever begin
            @(posedge clk);
            if (o_valid) begin
                if (^o_fx === 1'bx) begin
                    errors = errors + 1;
                    if (errors <= 20)
                        $display("X-OUTPUT idx=%0d in=%h", oidx, in_x[oidx]);
                end else if (o_fx !== exp[oidx]) begin
                    errors = errors + 1;
                    if (errors <= 20)
                        $display("MISMATCH idx=%0d in=%h exp=%h got=%h",
                                 oidx, in_x[oidx], exp[oidx], o_fx);
                end else begin
                    matched = matched + 1;
                end
                oidx = oidx + 1;
                if (oidx == MAXN) begin
                    if (errors == 0)
                        $display("TEST PASSED: %0d/%0d exact matches (mode=%0d, %s)",
                                 matched, MAXN, mode_arg, vecfile);
                    else
                        $display("TEST FAILED: %0d mismatches / %0d (mode=%0d, %s)",
                                 errors, MAXN, mode_arg, vecfile);
                    $finish;
                end
            end
        end
    end
endmodule
