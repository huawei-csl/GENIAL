// SPDX-License-Identifier: BSD-3-Clause Clear
// Copyright (c) 2024-2025 HUAWEI.
// All rights reserved.
//
// This software is released under the BSD 3-Clause Clear License.
// See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).


module AND (input A, input B, output Y);
 assign Y = A & B;
endmodule

module NAND (input A, input B, output Y);
 assign Y = ~(A & B);
endmodule

module OR (input A, input B, output Y);
 assign Y = A | B;
endmodule

module NOR (input A, input B, output Y);
 assign Y = ~(A | B);
endmodule

module NOT (input A, output Y);
 assign Y = ~A;
endmodule

module XOR (input A, input B, output Y);
 assign Y = A ^ B;
endmodule

module XNOR (input A, input B, output Y);
 assign Y = ~(A ^ B);
endmodule

module ANDNOT (input A, input B, output Y);
 assign Y = A & ~B;
endmodule

module ORNOT (input A, input B, output Y);
 assign Y = A | ~B;
endmodule

module MUX (input A, input B, input S, output Y);
 assign Y = (S == 0) ? A : B;
endmodule

module DFF_PN0 (
   input C,   // Clock input
   input D,   // Data input
   output Q,   // Output
   input R  // Active-low reset input
);
 reg q_int; // Internal register to hold the state

 always @(posedge C or negedge R) begin
   if (!R) begin         // Active-low reset
     q_int <= 1'b0;       // Reset Q to 0
   end else begin
     q_int <= D;          // Store D on positive clock edge
   end
 end

 assign Q = q_int;        // Continuous assignment to output
endmodule
