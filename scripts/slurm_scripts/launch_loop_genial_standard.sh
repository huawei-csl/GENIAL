#!/bin/bash
#SBATCH --job-name=top_genial_flowy
#SBATCH --time=600:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=AI-CPU
#SBATCH --nodelist=aisrv03
#SBATCH --output="/home/%u/genial_flowy_top_info.log"
#SBATCH --error="/home/%u/genial_flowy_top_error.log"

# Move to working directory
cd $HOME/proj/genial

# Activate Python Environment
# set -a
source .env
source $HOME/proj/genial/envs/313_genial/bin/activate
# set +a

# Launch Python command
# First launch:
python /home/$USER/proj/genial/scripts/loop_scripts/full_run_v2.py --is_slurm --config_filepath /home/$USER/proj/genial/scripts/loop_scripts/configs/flowy_loop_standard_emb.yaml --delete_merge_dirs --skip_init_gener
