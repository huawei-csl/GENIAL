# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

cd /app/tmp
tar -xzf /app/tmp/default.tar.gz
cp -r default ../design

cp mydesign_hdl/* ../design/hw/hdl/.
cp testbench.py ../design/test/.
cp mydesign_synth_wrapper.v ../design/hw/synth_wrapper/.

echo "======== Move Files to OpenRoad Flow Scripts ========"
cp -r /app/design/synth /prog/OpenROAD-flow-scripts/flow/designs/asap7/default_example
cp -r /app/design/hw/hdl /prog/OpenROAD-flow-scripts/flow/designs/src/default_example

echo "======== Run OpenROAD Synthesis ========"
cd /prog/OpenROAD-flow-scripts
make -C flow DESIGN_CONFIG=designs/asap7/default_example/config.mk synth
cp /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/1_synth.v /app/design/hw/synth_wrapper/mydesign_yosys.v

# echo "======== [Optional] Run OpenROAD Place and Route ========"
# make -C flow DESIGN_CONFIG=designs/asap7/default_example/config.mk route

echo "======== Simulate the Synthesized Design ========"
# If you want to perform SWACT, uncomment this block, but it will take a considerable amount of time:
# cd /app/tmp
# python /app/tmp/testbench_generator_standalone.py\
#     -t "/app/tmp/testbench_full.py.temp"\
#     -d "/app/tmp/mydesign_hdl/mydesign_comb.v"\
#     -s "/app/design/hw/synth_wrapper/mydesign_yosys.v"\
#     -o "/app/testbench"
# cp /app/testbench/testbench.py /app/design/test/.

# Run the testbench
cd /app/design/test
make

# Get per clock cycle activity
trace2power --clk-freq 1000000 --top i_mydesign_synthesized --limit-scope mydesign_synth_wrapper.i_mydesign_synthesized --remove-virtual-pins --output total_output dump.vcd
cp -r /app/design/test/total_output /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/

# trace2power --clk-freq 100000000 --top i_mydesign_synthesized --limit-scope mydesign_synth_wrapper.i_mydesign_synthesized --remove-virtual-pins --export-empty --output base_output dump.vcd
# cp /app/design/test/base_output /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/
# trace2power --clk-freq 100000000 --top i_mydesign_synthesized  --remove-virtual-pins --per-clock-cycle --only-glitches --clock-name clk_ci --output glitch_output dump.vcd
# cp -r /app/design/test/glitch_output /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/

# echo "======== Do Power Analysis (OLD) ========"
# cd /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/
# cp /app/design/power/sta_script.tcl /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/
# export LIB_DIR=/prog/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/
# sta sta_script.tcl

echo "======== Do Power Analysis (With Glitches) ========"
cd /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/
cp /app/design/power/power.tcl /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/
export LIB_DIR=/prog/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/
export LEF_DIR=/prog/OpenROAD-flow-scripts/flow/platforms/asap7/lef/
openroad -exit power.tcl

cp /app/design/power/recompose_total_power.py /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/
python3 recompose_total_power.py
# export LIB_DIR=/prog/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/
# sta sta_script.tcl


echo "======= Gathering Result Files ======="

mkdir /app/tmp/synth_results

cp /prog/OpenROAD-flow-scripts/flow/reports/asap7/default_example/base/synth_check.txt /app/tmp/synth_results/. 
cp /prog/OpenROAD-flow-scripts/flow/reports/asap7/default_example/base/synth_stat.txt /app/tmp/synth_results/. 
cp /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/result/total_output.rpt /app/tmp/synth_results/post_synth_power.rpt 
cp /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/post_synth_power.pqt /app/tmp/synth_results/. 
cp /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/post_synth_delay.rpt /app/tmp/synth_results/.
cp /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/post_route_power.rpt /app/tmp/synth_results/.
cp /prog/OpenROAD-flow-scripts/flow/results/asap7/default_example/base/1_synth.v /app/tmp/synth_results/mydesign_yosys.v
cp /app/design/test/results_*_db.pqt /app/tmp/synth_results/.
chmod -R 777 /app/tmp/synth_results
