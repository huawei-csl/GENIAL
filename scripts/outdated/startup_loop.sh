#!/bin/bash

# Name of the tmux session

loop_mode="normal"
session_name="loop_normal_unsigned"
device="2"
# yml_config_path="output/multiplier_2bi_4bo_permuti_allcells_unsigned_notech_fullsweep_only/optuna_configs/model_config_version216.yml"
yml_config_path="$WORK_DIR/output/multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only/optuna_configs/model_config_version216.yml"
loop_config="mult4b_unsigned_trans_minmax"
is_control_run=""

# Function to display usage
usage() {
    echo "Usage: $0  --device <value> --loop_mode <value> --experiment_name <value> --session_name <value> --yml_config_path <value>"
    exit 1
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --loop_mode) loop_mode="$2"; shift ;;
        --session_name) session_name="$2"; shift ;;
        --device) device="$2"; shift ;;
        --yml_config_path) yml_config_path="$2"; shift ;;
        --loop_config) loop_config="$2"; shift ;;
        --is_control_run) is_control_run="--is_control_run"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if the session already exists
tmux has-session -t $session_name 2>/dev/null

if [ $? != 0 ]; then
    # Create new tmux session
    tmux new-session -d -s $session_name

    tmux send-keys "source .env && 312_activate" C-m
    tmux send-keys "run_looper --config $loop_config --device $device --nb_workers 20 --yml_config_path $yml_config_path $is_control_run --synth_only" C-m

    # Attach to the tmux session
    tmux attach-session -t $session_name
else
    echo "Session $session_name already exists."
    exit 1
fi
