
import os
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'


# data_dir = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_260205/'

# res_list = os.listdir(data_dir)
#
suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()

count_dic = {}

for i, d in enumerate(suc_list):
    if i % 100 == 0:
        print(i)
    count_dic[d] = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0]


# count_dic2 = {d: c for d, c in count_dic.items() if c == 12}
#
# count_dic3 = {d: c for d, c in count_dic.items() if c != 12}

# success_dir = list(count_dic2.keys())
# success_dir.sort()
#
# unsuccess_dir = list(count_dic3.keys())
# unsuccess_dir.sort()


print(f"Completed: {sum([c == 6 for c in count_dic.values()])}")
print(f"Incompleted: {sum([c != 6 for c in count_dic.values()])}")

# print(f"Completed: {sum([c == 12 for c in count_dic.values()])}")
# print(f"Incompleted: {sum([c != 12 for c in count_dic.values()])}")

# d = suc_list[0]

print(pd.Series(count_dic.values()).value_counts())

# to_delete = [f for f in os.listdir(data_dir) if f not in count_dic or count_dic[f] < 5]
#
#
# counter = 0
# for d in to_delete:
#     try:
#         shutil.rmtree(f'{data_dir}{d}/')
#         print(d)
#     except:
#         print(d)
#     counter += 1
# print(f"deleted {counter} files")


new_set = set(os.listdir(data_dir))

old_data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_cache_260316/'

old_set = set(os.listdir(old_data_dir))


data_dir = old_data_dir

len(old_set & new_set)


count_dic_high = {k: v for k, v in count_dic.items() if v > 7}


df = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_cache_260316/res_00000000000051/flowy_data_record.parquet')

#
# for d in os.listdir(old_data_dir):
#     if d in new_set:
#         df_old = pd.read_parquet(f'{old_data_dir}{d}/flowy_data_record.parquet')
#         df_new = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
#         df = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(['run_identifier', 'step']).reset_index(drop=True)
#         df.to_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
#     else:
#         shutil.copytree(src=f'{old_data_dir}{d}', dst=f'{data_dir}{d}')
#

gene_all_dir = "/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out_FULL/"
gen_set = set(os.listdir(gene_all_dir))
gen_dst_dir = "/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/"

new_list = os.listdir(data_dir)
for d in new_list:
    if d in gen_set:
        shutil.copytree(src=f'{gene_all_dir}{d}', dst=f'{gen_dst_dir}{d}')
    else:
        print(f"Skipped: {d}")


for d in os.listdir(old_data_dir):
    if d in new_set:
        df_old = pd.read_parquet(f'{old_data_dir}{d}/flowy_data_record.parquet')
        df_new = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
        df = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(['run_identifier', 'step']).reset_index(drop=True)
        df.to_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
    else:
        shutil.copytree(src=f'{old_data_dir}{d}', dst=f'{data_dir}{d}')


data_dir

time_list = []
len_list = []

for d in os.listdir(old_data_dir):
    df = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
    time_list += df[df['step'] == 0]['runtime_full_mockturtle_step'].tolist()
    len_list.append(df.shape[0])
















