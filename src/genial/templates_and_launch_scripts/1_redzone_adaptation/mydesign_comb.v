// SPDX-License-Identifier: BSD-3-Clause Clear
// Copyright (c) 2024-2025 HUAWEI.
// All rights reserved.
//
// This software is released under the BSD 3-Clause Clear License.
// See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).


module mydesign_comb #(
    parameter int N_IN = 3, parameter int N_OUT = 6
  ) (
    input logic [N_IN-1:0] operand_a_i,
    input logic [N_IN-1:0] operand_b_i,
    output logic [N_OUT-1:0] result_o
  );

  logic [N_OUT-1:0] out;
  logic [2*N_IN-1:0] sel;

  assign sel[2*N_IN-1:N_IN] = operand_a_i;
  assign sel[N_IN-1:0] = operand_b_i;

  always @(sel) begin
    unique case(sel)
      6'b100100 : out = 6'b010000;
      6'b100101 : out = 6'b001100;
      6'b100110 : out = 6'b001000;
      6'b100111 : out = 6'b000100;
      6'b100000 : out = 6'b000000;
      6'b100001 : out = 6'b111100;
      6'b100010 : out = 6'b111000;
      6'b100011 : out = 6'b110100;
      6'b101100 : out = 6'b001100;
      6'b101101 : out = 6'b001001;
      6'b101110 : out = 6'b000110;
      6'b101111 : out = 6'b000011;
      6'b101000 : out = 6'b000000;
      6'b101001 : out = 6'b111101;
      6'b101010 : out = 6'b111010;
      6'b101011 : out = 6'b110111;
      6'b110100 : out = 6'b001000;
      6'b110101 : out = 6'b000110;
      6'b110110 : out = 6'b000100;
      6'b110111 : out = 6'b000010;
      6'b110000 : out = 6'b000000;
      6'b110001 : out = 6'b111110;
      6'b110010 : out = 6'b111100;
      6'b110011 : out = 6'b111010;
      6'b111100 : out = 6'b000100;
      6'b111101 : out = 6'b000011;
      6'b111110 : out = 6'b000010;
      6'b111111 : out = 6'b000001;
      6'b111000 : out = 6'b000000;
      6'b111001 : out = 6'b111111;
      6'b111010 : out = 6'b111110;
      6'b111011 : out = 6'b111101;
      6'b000100 : out = 6'b000000;
      6'b000101 : out = 6'b000000;
      6'b000110 : out = 6'b000000;
      6'b000111 : out = 6'b000000;
      6'b000000 : out = 6'b000000;
      6'b000001 : out = 6'b000000;
      6'b000010 : out = 6'b000000;
      6'b000011 : out = 6'b000000;
      6'b001100 : out = 6'b111100;
      6'b001101 : out = 6'b111101;
      6'b001110 : out = 6'b111110;
      6'b001111 : out = 6'b111111;
      6'b001000 : out = 6'b000000;
      6'b001001 : out = 6'b000001;
      6'b001010 : out = 6'b000010;
      6'b001011 : out = 6'b000011;
      6'b010100 : out = 6'b111000;
      6'b010101 : out = 6'b111010;
      6'b010110 : out = 6'b111100;
      6'b010111 : out = 6'b111110;
      6'b010000 : out = 6'b000000;
      6'b010001 : out = 6'b000010;
      6'b010010 : out = 6'b000100;
      6'b010011 : out = 6'b000110;
      6'b011100 : out = 6'b110100;
      6'b011101 : out = 6'b110111;
      6'b011110 : out = 6'b111010;
      6'b011111 : out = 6'b111101;
      6'b011000 : out = 6'b000000;
      6'b011001 : out = 6'b000011;
      6'b011010 : out = 6'b000110;
      6'b011011 : out = 6'b001001;
    endcase
  end


  assign result_o = out;

endmodule

// Input Value Encoding
// -4 -> 100
// -3 -> 101
// -2 -> 110
// -1 -> 111
// 0 -> 000
// 1 -> 001
// 2 -> 010
// 3 -> 011

// Output Value Encoding
// -12 -> 110100
// -9 -> 110111
// -8 -> 111000
// -6 -> 111010
// -4 -> 111100
// -3 -> 111101
// -2 -> 111110
// -1 -> 111111
// 0 -> 000000
// 1 -> 000001
// 2 -> 000010
// 3 -> 000011
// 4 -> 000100
// 6 -> 000110
// 8 -> 001000
// 9 -> 001001
// 12 -> 001100
// 16 -> 010000
