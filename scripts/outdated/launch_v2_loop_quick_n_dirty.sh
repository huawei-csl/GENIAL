#!/bin/bash

set -e

while true
do
   
    python src/genial/prototype_generator_v2.py \
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --output_dir_name loop_v2 \
    --do_prototype_pattern_gen \
    --device 3 \
    --nb_workers 48 \
    --synth_only \
    --nb_gener_prototypes 57344 \
    --trainer_version_number 86 \
    --batch_size 8192 &
    
    python src/genial/prototype_generator_v2.py \
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --output_dir_name loop_v2 \
    --do_prototype_pattern_gen \
    --device 1 \
    --nb_workers 48 \
    --synth_only \
    --nb_gener_prototypes 65536 \
    --trainer_version_number 86 \
    --batch_size 8192 &
    
    python src/genial/prototype_generator_v2.py \
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --output_dir_name loop_v2 \
    --do_prototype_pattern_gen \
    --device 0 \
    --nb_workers 48 \
    --synth_only \
    --nb_gener_prototypes 73728 \
    --trainer_version_number 86 \
    --batch_size 8192 

    task_launcher \
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --output_dir_name loop_v2 \
    --synth_only \
    --restart \
    --nb_workers 48 \
    --synth_only \
    --send_email
    
    task_analyzer --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --output_dir_name loop_v2 \
    --synth_only \
    --continue \
    --nb_workers 128 \
    --synth_only \
    --send_email

    task_trainer --output_dir_name loop_v2 \
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --batch_size 4096 \
    --max_epochs 100 \
    --check_val_every_n_epoch 1 \
    --device 3 \
    --nb_workers 64 \
    --score_type "trans" \
    --score_rescale_mode "standardize" \
    --synth_only \
    --finetune \
    --trainer_version_number 86 \
    --checkpoint_naming_style "enforce_increase" \
    --send_email

done
