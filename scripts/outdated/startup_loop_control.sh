#!/bin/bash

# Name of the tmux session
SESSION_NAME="loop_control"
# study_id="obj_r2nvalloss_early_valloss_500epochs" 

# Check if the session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    # Create new tmux session
    tmux new-session -d -s $SESSION_NAME

    tmux send-keys "source .env && 312_activate" C-m
    tmux send-keys "run_looper --experiment_name multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only --output_dir_name 40K_SYNTHV0_looptest_control --skip_init_analysis --skip_init_launching --config mult4b_trans_minmax --is_control_run --nb_workers 64" C-m

    # Attach to the tmux session
    tmux attach-session -t $SESSION_NAME
else
    echo "Session $SESSION_NAME already exists."
    tmux attach-session -t $SESSION_NAME
fi