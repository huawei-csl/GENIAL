# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

WORK_DIR=/app/oss_eda_flow_scripts
SYNTH_DIR=$WORK_DIR/synth

### Fixed files
# Pickle
mv /app/tmp/Bender.yml $WORK_DIR/Bender.yml
mv /app/tmp/mydesign.mk $WORK_DIR/mydesign.mk
# Synthesis
mv /app/tmp/abc.script $SYNTH_DIR/scripts/abc.script
mv /app/tmp/mydesign_notech_synth.ys $SYNTH_DIR/scripts/mydesign_notech_synth.ys
mv /app/tmp/project-synth.mk $SYNTH_DIR/project-synth.mk

### Generated files
mkdir -p $WORK_DIR/hw/mydesign/hdl
mv /app/tmp/mydesign_hdl/* $WORK_DIR/hw/mydesign/hdl/.

cd oss_eda_flow_scripts
make pickle
python py_scripts/mydesign_remove_pickle_suffix.py --top_name mydesign_top

cd synth
yosys -c scripts/mydesign_notech_synth.ys -l yosys_run.log

mkdir /app/tmp/synth_results
mkdir /app/tmp/synth_results/logs

cp $WORK_DIR/synth/*.log /app/tmp/synth_results/logs/.
cp $WORK_DIR/synth/reports/* /app/tmp/synth_results/.
cp $WORK_DIR/synth/out/mydesign_yosys.v /app/tmp/synth_results/.
touch /app/tmp/synth_results/synth_version_0
chmod -R 777 /app/tmp/synth_results
