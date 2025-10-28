// SPDX-License-Identifier: BSD-3-Clause Clear
// Copyright (c) 2024-2025 HUAWEI.
// All rights reserved.
//
// This software is released under the BSD 3-Clause Clear License.
// See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).


module mydesign_synth_wrapper #(
    parameter int N_IN = 3,
    parameter int N_OUT = 6
    ) (
    input logic clk_ci,
    input logic rst_ni,
    input logic [N_IN-1:0] operand_a_i,
    input logic [N_IN-1:0] operand_b_i,
    output logic [N_OUT-1:0] result_o
);

mydesign_top i_mydesign_synthesized(
    .clk_ci(clk_ci),
    .rst_ni(rst_ni),
    .operand_a_i_0_(operand_a_i[0]),
    .operand_a_i_1_(operand_a_i[1]),
    .operand_a_i_2_(operand_a_i[2]),
    .operand_b_i_0_(operand_b_i[0]),
    .operand_b_i_1_(operand_b_i[1]),
    .operand_b_i_2_(operand_b_i[2]),
    .result_o_0_(result_o[0]),
    .result_o_1_(result_o[1]),
    .result_o_2_(result_o[2]),
    .result_o_3_(result_o[3]),
    .result_o_4_(result_o[4]),
    .result_o_5_(result_o[5])
    );

endmodule
