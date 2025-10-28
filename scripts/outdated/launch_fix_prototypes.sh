# python prototype_generator_v2.py \
#     --do_prototype_pattern_gen \
#     --dst_output_dir_name loop_v2_fix_strategy3 \
#     --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
#     --output_dir_name loop_v2 \
#     --prototype_data_path output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/loop_v2/analysis_out/prototype_data_65536_2024-10-04_13-55_.npz

python prototype_generator_v2.py \
    --do_prototype_pattern_gen \
    --dst_output_dir_name loop_v2_fix_strategy3 \
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --output_dir_name loop_v2 \
    --prototype_data_path $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/loop_v2/analysis_out/prototype_data_65535_2024-10-09_07-04_.npz

python prototype_generator_v2.py \
    --do_prototype_pattern_gen \
    --dst_output_dir_name loop_v2_fix_strategy3 \
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --output_dir_name loop_v2 \
    --prototype_data_path $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/loop_v2/analysis_out/prototype_data_65536_2024-10-05_18-19_.npz

python prototype_generator_v2.py \
    --do_prototype_pattern_gen \
    --dst_output_dir_name loop_v2_fix_strategy3 \
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --output_dir_name loop_v2 \
    --prototype_data_path $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/loop_v2/analysis_out/prototype_data_65536_2024-10-06_21-42_.npz


task_launcher \
    --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --output_dir_name loop_v2_fix_strategy3 \
    --synth_only \
    --restart \
    --nb_workers 48 \
    --synth_only \
    --send_email

task_analyzer --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only \
    --output_dir_name loop_v2_fix_strategy3 \
    --synth_only \
    --continue \
    --nb_workers 128 \
    --synth_only \
    --send_email
    