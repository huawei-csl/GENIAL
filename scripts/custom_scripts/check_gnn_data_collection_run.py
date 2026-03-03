
import os
import shutil

def check_dir_completion(enc_dir):
    return len([f for f in os.listdir(enc_dir) if f.startswith('run_')]) == 6

# Root dir
data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/gnn_data_collection/synth_out/'

# All subdir
dir_list = os.listdir(data_dir)

# Completed subdir
comp_dir_list = [d for d in dir_list if check_dir_completion(data_dir + d)]

# Incomplete subdir
to_delete = [d for d in dir_list if not check_dir_completion(data_dir + d)]

# Delete incomplete subdir
for d in to_delete:
    shutil.rmtree(data_dir + d)






