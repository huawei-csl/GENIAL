#!/bin/bash

# Function to log messages with timestamps
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to clean up specific files and directories
clean_files_up() {
    local work_dir=$1

    log_message "Deleting files in $work_dir"

    rm -rf "$work_dir/analysis_out/plots_loop_analysis"
    rm -rf "$work_dir/analysis_out/plots"
    rm -rf "$work_dir/recommender_out"
    rm -rf "$work_dir/logs"
    rm -rf "$work_dir/synth_out"
    rm -rf "$work_dir/generation_out"
    rm -rf "$work_dir/test_out"
    rm -f  "$work_dir/valid_designs.db.pqt"
    rm -f  "$work_dir/encodings_dicts.db.pqt"
    rm -f  "$work_dir/launcher_run_from_scratch_info.txt"
    rm -f  "$work_dir/special_designs.json"
    rm -f  "$work_dir/merge_info.pqt"

    log_message "Cleanup complete."
}

# Run the function if arguments are provided
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <work_dir>"
    exit 1
fi

clean_files_up "$1"
