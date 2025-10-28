#!/bin/bash

start=`date +%s`

pushd synth/flowy
# The seed list is configured by the "best_seeds_example.json" file
run_get_best --env_option local_inside_docker --skip_tb --experiment synthesis_v4 --mockturtle_iterations 10 --mockturtle_nb_restarts 200 --nb_runs 50 > flowy_log.log

cp output/best_design/synthesized_design_synthesis_report.json ../out/mydesign_area_logic.rpt.json
cp output/best_design/synthesized_design_synthesis.v ../out/mydesign_yosys.v

python flowy/flows/analysis/get_best_seeds_of_exp.py --experiment synthesis_v4 --get_all_seeds
python /app/oss_eda_flow_scripts/synth/flowy/flowy/flows/analysis/run_analysis_flow.py --experiment_name synthesis_v4 --output_path output_plots

popd
python py_scripts/mydesign_replace_notech_cell_names.py

end=`date +%s`

runtime=$((end-start))
echo "Total runtime of synthesis: ${runtime}s"