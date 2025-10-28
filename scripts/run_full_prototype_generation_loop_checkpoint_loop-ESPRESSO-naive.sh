#!/bin/bash

# Load the .env
source .env

# Activate the conda environment
source activate design_gen

# Set input variables
INITIAL_OUT_DIR="10k_random_seed_42"
RUN_TYPE_SUFFIX="main_method_no_vae_espresso_random_checkpoint"
NB_PROTOTYPES=10000
DEVICE_ID=1
CONFIG_FILE="model_config_sample_w_cls_n_novae_ft.yml"
TASK="enc_to_score"


## Set the first prototype dir
#GEN_1_DIR="10k_random_seed_43"
#
## Set the first merge dir
#MERGE_1_DIR="${GEN_1_DIR}_espresso_naive_gen_new_2"

# Set the first prototype dir
GEN_1_DIR="10k_random_seed_42_main_method_no_vae_espresso_rn3_merge_1_13"

# Set the second prototype dir
GEN_2_DIR="AUG_10k_random_seed_42_main_method_no_vae_espresso_rn3_merge_2_13"

# Set the third prototype dir
GEN_3_DIR="AUG_10k_random_seed_42_main_method_no_vae_espresso_rn3_merge_3_13"

# Set the fourth prototype dir
GEN_4_DIR="AUG_10k_random_seed_42_main_method_no_vae_espresso_rn3_merge_4_13"

# Set the first merge dir
MERGE_1_DIR="${GEN_1_DIR}_naive_gen3"

# Set the second prototype dir
MERGE_2_DIR="${GEN_2_DIR}_naive_gen3"

# Set the third prototype dir
MERGE_3_DIR="${GEN_3_DIR}_naive_gen3"

# Set the fourth prototype dir
MERGE_4_DIR="${GEN_4_DIR}_naive_gen3"


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

    # Generate prototypes
    run_python_script $SRC_DIR/src/genial/utils/prototype_generator_v2-naive.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --do_prototype_pattern_gen \
        --output_dir_name $OUT_DIR \
        --yml_config_path $SRC_DIR/src/genial/templates_and_launch_scripts/$CONFIG_FILE \
        --dst_output_dir_name $GEN_DIR \
        --trainer_version_number 1 \
        --device $DEVICE_ID \
        --skip_swact \
        --skip_power \
        --ignore_user_prompts \
        --nb_workers 60 \
        --score_type complexity \
        --nb_gener_prototypes $NB_PROTOTYPES \
        --batch_size 5000

    log_message "Prototype generation completed..."

    # Synth prototypes
    run_python_script $SRC_DIR/src/genial/experiment/task_launcher.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --output_dir_name $GEN_DIR \
        --skip_swact \
        --skip_power \
        --ignore_user_prompts \
        --nb_workers 60 \
        --rebuild_db
    log_message "Prototype synth completed..."

    # Analyse prototypes
    run_python_script $SRC_DIR/src/genial/experiment/task_analyzer.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --output_dir_name $GEN_DIR \
        --skip_swact \
        --skip_power \
        --ignore_user_prompts \
        --nb_workers 60 \
        --rebuild_db


    log_message "Prototype Analysis completed..."

    log_message "Round completed."
}


## for i in 1 2 3 4 5 6 7 8 9 10 11; do
#for i in 1 2 3; do
#  ckpt_path="${!ckpt_var}"

# Round 1
main_pipeline "${GEN_1_DIR}" "${MERGE_1_DIR}"

## Round 2
#main_pipeline "${GEN_2_DIR}" "${MERGE_2_DIR}"
#
## Round 3
#main_pipeline "${GEN_3_DIR}" "${MERGE_3_DIR}"
#
## Round 4
#main_pipeline "${GEN_4_DIR}" "${MERGE_4_DIR}"
##done


