// tb_tars_bp_bb72_hz.sv - self-checking testbench for tars_bp_bb72_hz; run with any SystemVerilog-2012 simulator.
`timescale 1ns/1ps
module tb_tars_bp_bb72_hz;
  logic clk = 0, rst = 1, start = 0;
  logic [35:0] syndrome;
  logic busy, done, converged;
  logic [7:0] iters;
  logic [71:0] hard;
  logic [1151:0] post_flat;
  logic [35:0] vin [0:999];
  logic [1231:0] vout [0:999];
  tars_bp_bb72_hz dut (.*);
  always #2 clk = ~clk;
  integer i, errors = 0, cycles;
  initial begin
    $readmemh("tars_bp_bb72_hz_in.mem", vin);
    $readmemh("tars_bp_bb72_hz_out.mem", vout);
    repeat (3) @(posedge clk);
    rst = 0;
    for (i = 0; i < 1000; i = i + 1) begin
      @(negedge clk); syndrome = vin[i]; start = 1;
      @(negedge clk); start = 0; cycles = 1;
      while (!done) begin @(negedge clk); cycles = cycles + 1; end
      if ({iters, hard, post_flat} !== vout[i]) begin errors = errors + 1; $display("MISMATCH vector %0d", i); end
      if (cycles !== 2 * iters + 2) begin errors = errors + 1; $display("CYCLES vector %0d: %0d", i, cycles); end
    end
    $display("%0d vectors, %0d errors", 1000, errors);
    $finish;
  end
endmodule
