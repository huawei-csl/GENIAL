#!/bin/bash

# Initialize variables
device=""
study_id=""
output_dir_name=""
experiment_name=""

# Function to display usage
usage() {
    echo "Usage: $0 --device <value> --study_id <value>"
    exit 1
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device) device="$2"; shift ;;
        --study_id) study_id="$2"; shift ;;
        --output_dir_name) output_dir_name="$2"; shift ;;
        --experiment_name) experiment_name="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if both device and study_id are provided
if [ -z "$device" ] || [ -z "$study_id" ]; then
    echo "Error: Both --device and --study_id are required."
    usage
fi

# Echo the parameters
echo "device: $device"
echo "study_id: $study_id"

python src/genial/training/mains/optuna_trainer.py --output_dir_name $output_dir_name --experiment_name $experiment_name --device $device --study_id $study_id --batch_size 512 --nb_workers 20 --score_type "trans" --synth_only
