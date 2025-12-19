#!/bin/bash
#SBATCH --job-name=create_initial_dir
#SBATCH --output=tar_large_dir.out
#SBATCH --error=tar_large_dir.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=256
#SBATCH --nodelist=aisrv03
#SBATCH --reservation=ai-team
#SBATCH --partition=AI-CPU

cd /home/ramaudruz/proj/GENIAL

source .env
source ../GENIAL/envs/313_genial/bin/activate


# Using parallel gzip to speed up compression
python src/genial/utils/prototype_generator_from_csv.py --experiment_name multiplier_4bi_8bo_permuti_flowy --output_dir_name flowy_trans_run_12chains_3000steps_gen_iter0 --dst_output_dir_name uniform_initial_dir --trainer_version_number 0 --nb_gener_prototypes 40 --batch_size 20 --device 0 --yml_config_path resources/pretrained_model/embedding/model_config.yml --max_epochs 160 --score_type trans --nb_workers 60 --bulk_flow_dirname synth_out --do_prototype_pattern_gen --skip_swact --skip_power --skip_cmplx --ignore_user_prompts --keep_not_valid
