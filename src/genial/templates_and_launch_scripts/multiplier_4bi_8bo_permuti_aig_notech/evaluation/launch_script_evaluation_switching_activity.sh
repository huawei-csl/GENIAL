# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

WORK_DIR=/app/oss_eda_flow_scripts

mkdir -p ${WORK_DIR}"/synth/out"
mkdir -p ${WORK_DIR}"/hw/mydesign/tests"
mkdir -p ${WORK_DIR}"/hw/mydesign/hdl"

mv /app/tmp/mydesign_hdl/* $WORK_DIR/hw/mydesign/hdl/.
mv /app/tmp/mydesign_synth_wrapper.v $WORK_DIR/hw/mydesign/hdl/.
mv /app/tmp/notech_post_synth_swact_measure_tb.mk $WORK_DIR/hw/mydesign/tests/Makefile
mv /app/tmp/testbench.py $WORK_DIR/hw/mydesign/tests/.
mv /app/tmp/mydesign_yosys.v $WORK_DIR/synth/out/mydesign_yosys.v

cd oss_eda_flow_scripts

cd hw/mydesign/tests
# sleep infinity
make

mkdir /app/tmp/synth_results

ls $WORK_DIR/hw/mydesign/tests > /app/tmp/synth_results/ll.log
cp $WORK_DIR/hw/mydesign/tests/*_db.csv /app/tmp/synth_results/.
