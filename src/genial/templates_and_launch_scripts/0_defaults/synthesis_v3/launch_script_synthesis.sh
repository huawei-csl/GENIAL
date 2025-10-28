# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

ORIGIN_SRC_DIR=/app/original_design
WORK_DIR=/app/oss_eda_flow_scripts
SYNTH_DIR=$WORK_DIR/synth
FLOWY_DIR=$SYNTH_DIR/flowy

### Fixed files
mv /app/tmp/launch_synth_flow_v3p4_shortseedsearch.sh $WORK_DIR/launch_synth_flow_v3.sh
mv /app/tmp/get_best.py $WORK_DIR/py_scripts/get_best.py
cp /app/tmp/launcher_seeds.json $FLOWY_DIR/launcher_seeds.json

### Generated files
mkdir -p $ORIGIN_SRC_DIR/hw/hdl
cp /app/tmp/mydesign_hdl/* $ORIGIN_SRC_DIR/hw/hdl/.
cp $ORIGIN_SRC_DIR/hw/hdl/mydesign_comb.v $FLOWY_DIR/resources/sources/mydesign_comb.v.template 
cp $ORIGIN_SRC_DIR/hw/hdl/mydesign_top.v $FLOWY_DIR/resources/sources/mydesign_top.v.template 
# cp $ORIGIN_SRC_DIR/hw/hdl/mydesign_chip.sv $FLOWY_DIR/resources/sources/mydesign_chip.sv.template 

mkdir /app/tmp/synth_results
OUTPUT_DIR=/app/tmp/synth_results

cd $WORK_DIR
chmod +x ./launch_synth_flow_v3.sh

# sleep infinity
./launch_synth_flow_v3.sh > full_synth_flow.log

# Get Back Synthesized Design
cp $SYNTH_DIR/out/mydesign_area_logic.rpt.json $OUTPUT_DIR/mydesign_area_logic.rpt.json
cp $SYNTH_DIR/out/mydesign_yosys.v $OUTPUT_DIR/mydesign_yosys.v

# Get launcher seeds (not used for synth_v3p3)
cp /app/tmp/launcher_seeds.json $OUTPUT_DIR/launcher_seeds.json

# Get Runs Analysis Data
cp $FLOWY_DIR/run_execution_info.db.pqt $OUTPUT_DIR/run_execution_info.db.pqt
cp $FLOWY_DIR/output/db/synthesis_v3/analysis/seed_analysis/best_seeds/best_seeds.json $OUTPUT_DIR/nb_trans_and_seeds.json
cp $FLOWY_DIR/output_plots/transistor_count_plot.png $OUTPUT_DIR/transistor_count_plot.png
cp $FLOWY_DIR/output_plots/mockturtle_score_plot.png $OUTPUT_DIR/mockturtle_score_plot.png

# sleep infinity
touch $OUTPUT_DIR/synth_version_3p4
chmod -R 777 $OUTPUT_DIR

