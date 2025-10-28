#!/bin/bash

# Load the .env
source .env

# Activate the conda environment
source activate design_gen

# Set initial output dir
INITIAL_OUT_DIR="2025-03-13_09-52_002bcf7_SYNTHV0_200k_b"
# INITIAL_OUT_DIR="2025-03-13_09-52_002bcf7_SYNTHV0_1mio"

# Set the first prototype dir
GEN_1_DIR="${INITIAL_OUT_DIR}_novae_prototype_gen_1"

# Set the second prototype dir
GEN_2_DIR="${INITIAL_OUT_DIR}_novae_prototype_gen_2"

# Set the third prototype dir
GEN_3_DIR="${INITIAL_OUT_DIR}_novae_prototype_gen_3"

# Set the fourth prototype dir
GEN_4_DIR="${INITIAL_OUT_DIR}_novae_prototype_gen_4"

# Set the fifth prototype dir
GEN_5_DIR="${INITIAL_OUT_DIR}_novae_prototype_gen_5"

# Set the first merge dir
MERGE_1_DIR="${INITIAL_OUT_DIR}_novae_merge_1"

# Set the second prototype dir
MERGE_2_DIR="${INITIAL_OUT_DIR}_novae_merge_2"

# Set the third prototype dir
MERGE_3_DIR="${INITIAL_OUT_DIR}_novae_merge_3"

# Set the fourth prototype dir
MERGE_4_DIR="${INITIAL_OUT_DIR}_novae_merge_4"

# Set the fifth prototype dir
MERGE_5_DIR="${INITIAL_OUT_DIR}_novae_merge_5"

# Define log file with timestamp
LOG_FILE="$SRC_DIR/process_log_$(date '+%Y-%m-%d_%H-%M-%S').txt"

# Function to log messages with timestamps
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
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

main_pipeline() {
    local OUT_DIR=$1
    local GEN_DIR=$2
    local MERGE_DIR=$3

    log_message "Starting the round..."
    log_message "OUT_DIR=$OUT_DIR"
    log_message "GEN_DIR=$GEN_DIR"
    log_message "MERGE_DIR=$MERGE_DIR"


    if [[ $OUT_DIR == "2025-03-13_09-52_002bcf7_SYNTHV0_200k_b_novae_merge_3" ]]
    then
        echo "skipping synth"
    else
    # Synth initial dataset
    run_python_script $SRC_DIR/src/genial/experiment/task_launcher.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --output_dir_name $OUT_DIR \
        --skip_swact \
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
        --yml_config_path $SRC_DIR/src/genial/templates_and_launch_scripts/model_config_sample_w_cls_n_vae_ft.yml \
        --synth_only \
        --max_epochs 500 \
        --batch_size 1024 \
        --nb_workers 60 \
        --device 0 \
        --check_val_every_n_epoch 1 \
        --trainer_task finetune_from_ssl \
        --model_checkpoint $WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech/$OUT_DIR/trainer_out/lightning_logs/version_0/checkpoints/72_0.0031_000.ckpt

    log_message "Model finetuning completed..."
#
#    fi

    # Generate prototypes
    run_python_script $SRC_DIR/src/genial/utils/prototype_generator_v2.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --do_prototype_pattern_gen \
        --output_dir_name $OUT_DIR \
        --dst_output_dir_name $GEN_DIR \
        --trainer_version_number 1 \
        --device 0 \
        --synth_only \
        --nb_workers 60 \
        --nb_gener_prototypes 100000 \
        --batch_size 5000

    log_message "Prototype generation completed..."

    # Synth prototypes
    run_python_script $SRC_DIR/src/genial/experiment/task_launcher.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --output_dir_name $GEN_DIR \
        --skip_swact \
        --nb_workers 60 \
        --rebuild_db

    log_message "Prototype synth completed..."

    # Analyse prototypes
    run_python_script $SRC_DIR/src/genial/experiment/task_analyzer.py \
        --experiment_name multiplier_4bi_8bo_permuti_allcells_notech \
        --output_dir_name $GEN_DIR \
        --synth_only \
        --nb_workers 60 \
        --rebuild_db

    log_message "Prototype Analysis completed..."

    fi

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

    log_message "Round completed."
}

# Round 1
# main_pipeline "$INITIAL_OUT_DIR" "$GEN_1_DIR" "$MERGE_1_DIR"

# Round 2
# main_pipeline "$MERGE_1_DIR" "$GEN_2_DIR" "$MERGE_2_DIR"

# Round 3
# main_pipeline "$MERGE_2_DIR" "$GEN_3_DIR" "$MERGE_3_DIR"

# Round 4
main_pipeline "$MERGE_3_DIR" "$GEN_4_DIR" "$MERGE_4_DIR"

# Round 5
main_pipeline "$MERGE_4_DIR" "$GEN_5_DIR" "$MERGE_5_DIR"
