#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --output="/home/%u/slurm_logs/genial/sbatch_info/genial_flowy_%j_%N_$timestamp.log"
#SBATCH --error="/home/%u/slurm_logs/genial/sbatch_error/genial_flowy_%j_%N_$timestamp.log"
# Move to working directory
cd $$HOME/proj/GENIAL

# Activate Python Environment
# set -a
source .env
source $$HOME/envs/313_genial/bin/activate
# set +a

# Launch Python command
$cmd
