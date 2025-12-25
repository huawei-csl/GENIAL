
import os
import shutil
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd

# ref_df = pd.read_csv('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/misc/count_1_df_251224.csv')
#
# data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'
#
# to_remove = [f for f in os.listdir(data_dir) if f not in ref_df['k'].tolist()]
#
# to_keep = [f for f in os.listdir(data_dir) if f in ref_df['k'].tolist()]
#
# for f in to_remove:
#     shutil.rmtree(data_dir + f)

ref_df = pd.read_csv('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/misc/count_23_df_251224.csv')

data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'

to_remove = [f for f in os.listdir(data_dir) if f not in ref_df['k'].tolist()]

to_keep = [f for f in os.listdir(data_dir) if f in ref_df['k'].tolist()]

# for f in to_remove:
#     shutil.rmtree(data_dir + f)



###########


import os
import shutil
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd


data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'


res_list = os.listdir(data_dir)

suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()

count_dic = {d: pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0] for d in suc_list}

count_dic2 = {d: c for d, c in count_dic.items() if c == 11}

count_dic3 = {d: c for d, c in count_dic.items() if c != 11}

print(f"Completed: {sum([c == 11 for c in count_dic.values()])}")
print(f"Incompleted: {sum([c != 11 for c in count_dic.values()])}")
