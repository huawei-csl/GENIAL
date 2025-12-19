#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --output="/home/%u/slurm_logs/genial/sbatch_info/genial_flowy_%j_%N_20251219_101556.log"
#SBATCH --error="/home/%u/slurm_logs/genial/sbatch_error/genial_flowy_%j_%N_20251219_101556.log"
#SBATCH --reservation=ai-team
#SBATCH --nodelist=aisrv03
# Move to working directory
cd $HOME/proj/GENIAL

# Activate Python Environment
# set -a
source .env
source $HOME/envs/313_genial/bin/activate
# set +a

# Launch Python command
python src/genial/experiment/task_launcher.py --experiment_name multiplier_4bi_8bo_permuti_flowy --output_dir_name starting_dataset --nb_workers 36 --skip_swact --skip_power --skip_cmplx --bulk_flow_dirname synth_out --skip_swact --skip_power --ignore_user_prompts --keep_not_valid --design_number_list 00000000000000 0000000000001