#!/bin/bash

# Load the .env
source .env

# Activate the conda environment
source activate design_gen

# Set input variables
INITIAL_OUT_DIR="20k_random_seed_100"
RUN_TYPE_SUFFIX="main_method"
NB_PROTOTYPES=20000
DEVICE_ID=0
CONFIG_FILE="model_config_sample_w_cls_n_vae_ft.yml"
TASK="finetune_from_ssl"
SSL_CHECKPOINT="trainer_out/lightning_logs/version_0/checkpoints/114_0.0491_000.ckpt"
# SSL_CHECKPOINT="trainer_out/lightning_logs/version_0/checkpoints/74_0.0019_000.ckpt"


# INITIAL_OUT_DIR="2025-03-13_09-52_002bcf7_SYNTHV0_1mio"

# Set the first prototype dir
GEN_1_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_1"

# Set the second prototype dir
GEN_2_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_2"

# Set the third prototype dir
GEN_3_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_3"

# Set the fourth prototype dir
GEN_4_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_4"

# Set the fifth prototype dir
GEN_5_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_5"

# Set the sixth prototype dir
GEN_6_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_6"

# Set the seventh prototype dir
GEN_7_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_7"

# Set the eighth prototype dir
GEN_8_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_8"

# Set the nineth prototype dir
GEN_9_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_9"

# Set the tenth prototype dir
GEN_10_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_10"

# Set the eleventh prototype dir
GEN_11_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_11"

# Set the twelveth prototype dir
GEN_12_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_12"

# Set the thirteenth prototype dir
GEN_13_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_13"

# Set the fourteenth prototype dir
GEN_14_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_14"

# Set the fifteenth prototype dir
GEN_15_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_prototype_gen_15"

# Set the first merge dir
MERGE_1_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_1"

# Set the second prototype dir
MERGE_2_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_2"

# Set the third prototype dir
MERGE_3_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_3"

# Set the fourth prototype dir
MERGE_4_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_4"

# Set the fifth prototype dir
MERGE_5_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_5"

# Set the sixth merge dir
MERGE_6_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_6"

# Set the seventh prototype dir
MERGE_7_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_7"

# Set the eighth prototype dir
MERGE_8_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_8"

# Set the nineth prototype dir
MERGE_9_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_9"

# Set the tenth prototype dir
MERGE_10_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_10"

# Set the eleventh merge dir
MERGE_11_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_11"

# Set the twelveth prototype dir
MERGE_12_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_12"

# Set the thirteenth prototype dir
MERGE_13_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_13"

# Set the fourteenth prototype dir
MERGE_14_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_14"

# Set the fifteens prototype dir
MERGE_15_DIR="${INITIAL_OUT_DIR}_${RUN_TYPE_SUFFIX}_merge_15"


# Define log file with timestamp
LOG_FILE="$SRC_DIR/process_log_$(date '+%Y-%m-%d_%H-%M-%S').txt"

# Function to log messages with timestamps
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
}

# Function to clean up previous dirs
clean_files_up() {
    local work_dir=$1
    local rel_dir=$2

    log_message "Deleting files in $rel_dir"

    # Delete all files except checkpoints and synth analysis
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/analysis_out/plots_loop_analysis
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/analysis_out/plots
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/recommender_out
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/logs
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/synth_out
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/generation_out
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/test_out
    rm $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/valid_designs.db.pqt
    rm $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/encodings_dicts.db.pqt
    rm $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/launcher_run_from_scratch_info.txt
    rm $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/special_designs.json
    rm $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/merge_info.pqt

}

# Run python script with logs
run_python_script() {
    local script_path=$1
    shift  # Remove the first argument (script path) from the list, leaving only the arguments for the Python script

    log_message "Running Python script ($script_path)..."
    python $script_path "$@"
    if [ $? -eq 0 ]; then
        log_message "Python script executed successfully."
    else
        log_message "Error running Python script."
        exit 1
    fi
}

# Run python script with logs
clean_files_up() {
    local work_dir=$1
    local rel_dir=$2

    log_message "Deleting files in $rel_dir"

    # Delete all files except checkpoints and synth analysis
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/analysis_out/plots_loop_analysis
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/analysis_out/plots
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/recommender_out
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/logs
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/synth_out
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/generation_out
    rm -r $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/test_out
    rm $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/valid_designs.db.pqt
    rm $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/encodings_dicts.db.pqt
    rm $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/launcher_run_from_scratch_info.txt
    rm $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/special_designs.json
    rm $work_dir/output/multiplier_4bi_8bo_permuti_allcells_notech/$rel_dir/merge_info.pqt

}

main_pipeline() {
    local OUT_DIR=$1
    local GEN_DIR=$2
    local MERGE_DIR=$3

    log_message "Starting the round..."
    log_message "OUT_DIR=$OUT_DIR"
    log_message "GEN_DIR=$GEN_DIR"
    log_message "MERGE_DIR=$MERGE_DIR"


    # if [[ $OUT_DIR == "2025-03-13_09-52_002bcf7_SYNTHV0_200k" ]]
    # then
    #     echo "skipping synth"
    # else
    # Synth initial dataset
    run_python_script $SRC_DIR/src/genial/experiment/task_launcher.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --output_dir_name $OUT_DIR \
        --skip_swact \
        --skip_power \
        --nb_workers 60 \
        --rebuild_db

    log_message "Synth initial dataset completed..."

    # Analyse initial dataset
    run_python_script $SRC_DIR/src/genial/experiment/task_analyzer.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --output_dir_name $OUT_DIR \
        --synth_only \
        --nb_workers 60 \
        --rebuild_db

    log_message "Analyse initial dataset completed..."

    # Train model
    run_python_script $SRC_DIR/src/genial/training/mains/trainer_enc_to_score_value.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --output_dir_name $OUT_DIR \
        --yml_config_path $SRC_DIR/src/genial/templates_and_launch_scripts/$CONFIG_FILE \
        --synth_only \
        --max_epochs 500 \
        --batch_size 2000 \
        --nb_workers 60 \
        --device $DEVICE_ID \
        --check_val_every_n_epoch 1 \
        --trainer_task $TASK \
        --model_checkpoint $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech/$OUT_DIR/$SSL_CHECKPOINT

    log_message "Model finetuning completed..."

    # fi

    # Generate prototypes
    run_python_script $SRC_DIR/src/genial/utils/prototype_generator_v2.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --do_prototype_pattern_gen \
        --output_dir_name $OUT_DIR \
        --yml_config_path $SRC_DIR/src/genial/templates_and_launch_scripts/$CONFIG_FILE \
        --dst_output_dir_name $GEN_DIR \
        --trainer_version_number 1 \
        --device $DEVICE_ID \
        --synth_only \
        --nb_workers 60 \
        --nb_gener_prototypes $NB_PROTOTYPES \
        --batch_size 5000

    log_message "Prototype generation completed..."

    # Synth prototypes
    run_python_script $SRC_DIR/src/genial/experiment/task_launcher.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --output_dir_name $GEN_DIR \
        --skip_swact \
        --skip_power \
        --nb_workers 60 \
        --rebuild_db

    log_message "Prototype synth completed..."

    # Analyse prototypes
    run_python_script $SRC_DIR/src/genial/experiment/task_analyzer.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --output_dir_name $GEN_DIR \
        --synth_only \
        --skip_power \
        --nb_workers 60 \
        --rebuild_db

    log_message "Prototype Analysis completed..."

    # Merge prototypes with previous dataset
    run_python_script $SRC_DIR/src/genial/utils/merge_output_dirs.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --output_dir_name_0 $OUT_DIR \
        --output_dir_name_1 $GEN_DIR \
        --dst_output_dir_name $MERGE_DIR \
        --frac_0 1 \
        --frac_1 1

    log_message "Merging completed..."

    # Synth merge dir
    run_python_script $SRC_DIR/src/genial/experiment/task_launcher.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --nb_workers 60 \
        --output_dir_name $MERGE_DIR \
        --skip_swact \
        --skip_power \
        --rebuild_db

    log_message "Synth merge dir completed..."

    # Analyse merge dir
    run_python_script $SRC_DIR/src/genial/experiment/task_analyzer.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --nb_workers 60 \
        --output_dir_name $MERGE_DIR \
        --synth_only \
        --rebuild_db

    log_message "Analyse merge dir completed..."

    # Copy SSL checkpoint
    mkdir $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech/$MERGE_DIR/trainer_out/lightning_logs
    cp -r $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech/$OUT_DIR/trainer_out/lightning_logs/version_0 \
        $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech/$MERGE_DIR/trainer_out/lightning_logs

    log_message "Copy checkpoint completed..."

    # Out dir cleanup
    if [[ $OUT_DIR == $INITIAL_OUT_DIR ]]; then
        echo "keeping the original dataset"
    else
        clean_files_up "$WORK_DIR" "$OUT_DIR"

        log_message "Cleanup of dir $OUT_DIR completed..."
    fi

    # Gen dir cleanup
    clean_files_up "$WORK_DIR" "$GEN_DIR"
    log_message "Cleanup of dir $GEN_DIR completed..."

    log_message "Round completed."
}

# Round 1
# main_pipeline "$INITIAL_OUT_DIR" "$GEN_1_DIR" "$MERGE_1_DIR"

# Round 2
# main_pipeline "$MERGE_1_DIR" "$GEN_2_DIR" "$MERGE_2_DIR"

# Round 3
# main_pipeline "$MERGE_2_DIR" "$GEN_3_DIR" "$MERGE_3_DIR"

# Round 4
# main_pipeline "$MERGE_3_DIR" "$GEN_4_DIR" "$MERGE_4_DIR"

# Round 5
# main_pipeline "$MERGE_4_DIR" "$GEN_5_DIR" "$MERGE_5_DIR"

# Round 6
# main_pipeline "$MERGE_5_DIR" "$GEN_6_DIR" "$MERGE_6_DIR"

# Round 7
# main_pipeline "$MERGE_6_DIR" "$GEN_7_DIR" "$MERGE_7_DIR"

## Round 8
#main_pipeline "$MERGE_7_DIR" "$GEN_8_DIR" "$MERGE_8_DIR"
#
## Round 9
#main_pipeline "$MERGE_8_DIR" "$GEN_9_DIR" "$MERGE_9_DIR"
#
## Round 10
#main_pipeline "$MERGE_9_DIR" "$GEN_10_DIR" "$MERGE_10_DIR"

## Round 11
#main_pipeline "$MERGE_10_DIR" "$GEN_11_DIR" "$MERGE_11_DIR"
#
## Round 12
#main_pipeline "$MERGE_11_DIR" "$GEN_12_DIR" "$MERGE_12_DIR"
#
## Round 13
#main_pipeline "$MERGE_12_DIR" "$GEN_13_DIR" "$MERGE_13_DIR"
#
## Round 14
#main_pipeline "$MERGE_13_DIR" "$GEN_14_DIR" "$MERGE_14_DIR"
#
## Round 15
#main_pipeline "$MERGE_14_DIR" "$GEN_15_DIR" "$MERGE_15_DIR"
