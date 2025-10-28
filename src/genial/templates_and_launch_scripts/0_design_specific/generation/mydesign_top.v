// SPDX-License-Identifier: BSD-3-Clause Clear
// Copyright (c) 2024-2025 HUAWEI.
// All rights reserved.
//
// This software is released under the BSD 3-Clause Clear License.
// See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

module mydesign_top #(
    parameter int N_IN = 3,
    parameter int N_OUT = 6
  ) (
    input logic clk_ci,
    input logic rst_ni,
    input logic [N_IN-1:0] operand_a_i,
    input logic [N_IN-1:0] operand_b_i,
    output logic [N_OUT-1:0] result_o
);

logic [N_OUT-1:0] result_d;
logic [N_OUT-1:0] result_q;

(* dont_touch = "true" *) mydesign_comb #(
  .N_IN(N_IN),
  .N_OUT(N_OUT)
) i_mydesign_comb (
  .operand_a_i(operand_a_i),
  .operand_b_i(operand_b_i),
  .result_o(result_d)
);

always_ff @(posedge clk_ci, negedge rst_ni) begin
  if (~rst_ni) begin
    result_q <= 0;
  end else begin
    result_q <= result_d;
  end
end

assign result_o = result_q;

endmodule
