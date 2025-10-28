# task_launcher --experiment_name multiplier_3bi_6bo_permuti_allcells_notech --output_dir_name 2024-08-09_09-50_d2ecbf7 --restart --nb_workers 96
# task_launcher --experiment_name multiplier_3bi_6bo_permuti_allcells_notech --output_dir_name 2024-08-09_21-11_4a6e61e --restart --nb_workers 96

# task_analyzer --experiment_name multiplier_3bi_6bo_permuti_allcells_notech --output_dir_name 2024-08-09_21-11_4a6e61e --rebuild_db

# task_launcher --experiment_name multiplier_3bi_6bo_permuti_mig_notech --nb_workers 1 --design_number_list "99999" --restart
# task_analyzer --experiment_name multiplier_3bi_6bo_permuti_mig_notech --output_dir_name 2024-08-09_21-11_4a6e61e --design_number_list "99999"

# task_launcher --experiment_name multiplier_3bi_6bo_permuti_mig_notech --nb_workers 128 --do_gener --only_gener
# task_launcher --experiment_name multiplier_3bi_6bo_permuti_mig_noresynth_notech --nb_workers 128 --do_gener --only_gener
# task_launcher --experiment_name multiplier_3bi_6bo_permuti_allcells_noresynth_notech --nb_workers 128 --do_gener --only_gener

task_launcher --experiment_name multiplier_3bi_6bo_permuti_mig_notech --output_dir_name 2024-08-12_07-43_4a6e61e --nb_workers 96 --restart
task_analyzer --experiment_name multiplier_3bi_6bo_permuti_mig_notech --output_dir_name 2024-08-12_07-43_4a6e61e --nb_workers 128 --rebuild_db

task_launcher --experiment_name multiplier_3bi_6bo_permuti_mig_noresynth_notech --output_dir_name 2024-08-12_07-44_4a6e61e --nb_workers 96 --restart
task_analyzer --experiment_name multiplier_3bi_6bo_permuti_mig_noresynth_notech --output_dir_name 2024-08-12_07-44_4a6e61e --nb_workers 128 --rebuild_db

task_launcher --experiment_name multiplier_3bi_6bo_permuti_allcells_noresynth_notech --output_dir_name 2024-08-12_07-44_4a6e61e --nb_workers 96 --restart
task_analyzer --experiment_name multiplier_3bi_6bo_permuti_allcells_noresynth_notech --output_dir_name 2024-08-12_07-44_4a6e61e --nb_workers 128 --rebuild_db
