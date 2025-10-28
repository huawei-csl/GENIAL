#!/bin/bash

# Load the .env
source .env

# Activate the conda environment
source activate design_gen

# Set input variables
RUN_TYPE_SUFFIX="main_method_new_sampling"
NB_PROTOTYPES=10000
DEVICE_ID=0
CONFIG_FILE="model_config_sample_w_cls_n_vae_ft.yml"
TASK="finetune_from_ssl"
SSL_CHECKPOINT="trainer_out/lightning_logs/version_0/checkpoints/114_0.0491_000.ckpt"

# Set the initial dir
INITIAL_OUT_DIR1="20k_random_seed_100_main_method_merge_8"

# Set the first prototype dir
GEN_1_DIR="${INITIAL_OUT_DIR1}_${RUN_TYPE_SUFFIX}_prototype_gen_1"

# Set the initial dir
INITIAL_OUT_DIR2="20k_random_seed_100_main_method_merge_8_90pct_NEW"

# Set the first prototype dir
GEN_2_DIR="${INITIAL_OUT_DIR2}_${RUN_TYPE_SUFFIX}_prototype_gen_1"

# Set the initial dir
INITIAL_OUT_DIR3="20k_random_seed_100_main_method_merge_8_80pct_NEW"

# Set the first prototype dir
GEN_3_DIR="${INITIAL_OUT_DIR3}_${RUN_TYPE_SUFFIX}_prototype_gen_1"

# Set the initial dir
INITIAL_OUT_DIR4="20k_random_seed_100_main_method_merge_8_70pct_NEW"

# Set the first prototype dir
GEN_4_DIR="${INITIAL_OUT_DIR4}_${RUN_TYPE_SUFFIX}_prototype_gen_1"

# Set the initial dir
INITIAL_OUT_DIR5="20k_random_seed_100_main_method_merge_8_60pct_NEW"

# Set the first prototype dir
GEN_5_DIR="${INITIAL_OUT_DIR5}_${RUN_TYPE_SUFFIX}_prototype_gen_1"


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

    log_message "Starting the round..."
    log_message "OUT_DIR=$OUT_DIR"
    log_message "GEN_DIR=$GEN_DIR"

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

    log_message "Round completed."
}

# Round 1
# main_pipeline "$INITIAL_OUT_DIR1" "$GEN_1_DIR"

# Round 2
main_pipeline "$INITIAL_OUT_DIR2" "$GEN_2_DIR"

# Round 3
main_pipeline "$INITIAL_OUT_DIR3" "$GEN_3_DIR"

# Round 4
main_pipeline "$INITIAL_OUT_DIR4" "$GEN_4_DIR"

# Round 5
main_pipeline "$INITIAL_OUT_DIR5" "$GEN_5_DIR"


