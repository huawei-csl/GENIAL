#!/bin/bash

start=`date +%s`

pushd synth/flowy
# The seed list is configured by the "best_seeds_example.json" file
run_get_best --env_option local_inside_docker --skip_tb --use_seeds --seeds_filepath "launcher_seeds.json" --experiment run_known_seed_lists > flowy_log.log

cp output/best_design/synthesized_design_synthesis_report.json ../out/mydesign_area_logic.rpt.json
cp output/best_design/synthesized_design_synthesis.v ../out/mydesign_yosys.v
python flowy/flows/analysis/get_best_seeds_of_exp.py --experiment run_known_seed_lists --get_all_seeds

popd
python py_scripts/mydesign_replace_notech_cell_names.py

end=`date +%s`

runtime=$((end-start))
echo "Total runtime of synthesis: ${runtime}s"