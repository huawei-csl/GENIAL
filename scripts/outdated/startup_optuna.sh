#!/bin/bash

# Name of the tmux session
SESSION_NAME="optuna"
study_id="<study_id>" 

output_dir_name="<output_dir_name>"
experiment_name="multiplier_4bi_8bo_permuti_allcells_notech_normal_only"

# Check if the session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    # Create new tmux session
    tmux new-session -d -s $SESSION_NAME

    # Left column
    # x0 | y0
    tmux send-keys "source .env && 312_activate" C-m
    tmux send-keys "optuna-dashboard 'mysql://$USER@localhost/optuna_studies?unix_socket=/var/run/mysqld/mysqld.sock'" C-m
    tmux split-window -h

    # Now focus on the left column (pane 0) and split it horizontally into 3 panes
    tmux select-pane -t 0
    tmux split-window -v
    # x0 | y1
    
    tmux split-window -v
    # x0 | y2
    tmux send-keys "source .env && 312_activate && ./launch_optuna.sh --device 4 --study_id $study_id --output_dir_name $output_dir_name --experiment_name $experiment_name" C-m   # Third pane command

    # Now move to the right column (pane 3) and split it horizontally into 3 panes
    tmux select-pane -t 3
    # x1 | y0
    tmux send-keys "source .env && 312_activate && ./launch_optuna.sh --device 5 --study_id $study_id --output_dir_name $output_dir_name --experiment_name $experiment_name" C-m   # Fifth pane command
    
    tmux split-window -v
    # x1 | y1
    tmux send-keys "source .env && 312_activate && ./launch_optuna.sh --device 6 --study_id $study_id --output_dir_name $output_dir_name --experiment_name $experiment_name" C-m   # Fifth pane command

    tmux split-window -v
    # x1 | y2
    tmux send-keys "source .env && 312_activate && ./launch_optuna.sh --device 2 --study_id $study_id --output_dir_name $output_dir_name --experiment_name $experiment_name" C-m   # Fifth pane command
    
    # Uniformly distribute all panes
    tmux select-layout tiled
    
    # Uniformly distribute all panes
    tmux select-pane -t 0
    tmux split-window -h
    # Now we should be on the right side of panel 0
    # Add tensorboard runner
    tmux send-keys "source .env && 312_activate && tensorboard --logdir='$WORK_DIR/output/$experiment_name/$output_dir_name/trainer_out/optuna/$study_id/lightning_logs'" C-m

    # Attach to the tmux session
    tmux attach-session -t $SESSION_NAME
else
    echo "Session $SESSION_NAME already exists."
    tmux attach-session -t $SESSION_NAME
fi