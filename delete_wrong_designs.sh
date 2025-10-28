#!/bin/bash

python src/genial/prototype_generator_v2.py \
 --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
 --output_dir_name loop_v2 \
 --do_delete_wrong_designs

task_analyzer --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --output_dir_name loop_v2 \
    --synth_only \
    --continue \
    --nb_workers 128 \
    --synth_only \
    --send_email



