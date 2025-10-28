#!/bin/bash
#SBATCH --job-name=top_genial_flowy
#SBATCH --time=600:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=<partition>
#SBATCH --nodelist=<node>
#SBATCH --output="/home/%u/genial_flowy_top_info.log"
#SBATCH --error="/home/%u/genial_flowy_top_error.log"

# Move to working directory
cd $SRC_DIR

# Activate Python Environment
# set -a
source .env
source $HOME/envs/312_global/bin/activate
# set +a

# Launch Python command
python $SRC_DIR/scripts/loop_scripts/full_run_v2.py --is_slurm --config_filepath $SRC_DIR/scripts/loop_scripts/configs/flowy_loop_default.yaml
