#!/bin/bash

start=`date +%s`

pushd synth/flowy
# The iterations and nb_restarts are ignored for single seed list synthesis
python run_flow_test_incl_store_mc_trial.py --skip_generation --skip_testbench --mockturtle_iterations 10 --mockturtle_nb_restarts 200

popd
python py_scripts/mydesign_replace_notech_cell_names.py

end=`date +%s`

runtime=$((end-start))
echo "Total runtime of synthesis: ${runtime}s"