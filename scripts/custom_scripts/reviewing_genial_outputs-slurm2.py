
import os
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_dir = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'


# data_dir = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_260205/'

# res_list = os.listdir(data_dir)
#
suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()

# print(suc_list)


#
# base_dir = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/'
#
# len(os.listdir(base_dir + 'generation_out_FULL'))
#
# len(os.listdir(base_dir + 'generation_out_felix'))
#
#
# remaining_set = (
# (
#         set(os.listdir(base_dir + 'generation_out_FULL')) - set(os.listdir(base_dir + 'generation_out_felix'))
# ) - set(os.listdir(base_dir + 'synth_out_260205'))
# ) - set(os.listdir(base_dir + 'synth_out_260211'))
#
#
#
# for d in os.listdir(base_dir + 'generation_out_remaining_260212'):
#     if d not in remaining_set:
#         shutil.rmtree(base_dir + 'generation_out_remaining_260212/' + d)
#
# len(os.listdir(base_dir + 'generation_out_remaining_260212'))
#
#
# for d in os.listdir(base_dir + 'generation_out_remaining_260212_even'):
#     if int(d.split('_')[-1]) % 2 == 1:
#         shutil.rmtree(base_dir + 'generation_out_remaining_260212_even/' + d)




# file_set = set()
#
# for s in suc_list:
#     file_set.update(os.listdir(f'{data_dir}{s}'))


# df = pd.read_parquet('/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/res_00000000000000/flowy_data_record.parquet')


# count_list = [pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0] for d in suc_list]

# count_dic = {d: pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0] for d in suc_list}

#
# data_list = []
#
# counter_i = 0
#
# for i, d in enumerate(suc_list):
#     if i % 100 == 0:
#         print(i)
#     df = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
#     # run_time = df[df['step']==0]['runtime_full_mockturtle_step'].mean()
#     # data_list.append({'d': d, 'run_time': run_time})
#     # df['d'] = d
#     # data_list.append(
#     #     df[df['step']==0].reset_index(drop=True)
#     # )
#     cond = df['runtime_full_mockturtle_step'] > 1000
#
#     if df[cond]['run_identifier'].unique().shape[0] >= 5:
#         data_list.append(d)
#
# print(counter_i)
#
# diff = set(os.listdir(f'{data_dir}')) - set(data_list)
#
# diff_list = list(diff)
#
# gene_out_felix = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out_felix/'
# gene_out = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'
#
#
# for d in os.listdir(gene_out_felix):
#     if d not in diff:
#         shutil.rmtree(f'{gene_out_felix}{d}')




# len(os.listdir(gene_out_felix))
#
#
# len(os.listdir(gene_out))
# len(diff_list)
#
# df_conc = pd.concat(data_list, ignore_index=True).sort_values('runtime_full_mockturtle_step').reset_index(drop=True)
# anal10_df = pd.DataFrame(data_list).sort_values('run_time').reset_index(drop=True)


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

pd.Series(count_dic.values()).value_counts()

counter = 0
for d in os.listdir(data_dir):
    if d not in count_dic:
        try:
            shutil.rmtree(f'{data_dir}{d}/')
            print(d)
        except:
            print(d)
        counter += 1
print(f"deleted {counter} files")



# count_dic1 = {k: v for k, v in count_dic.items() if v ==1}
#
# sub_dir = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out_subset/'
#
# for d in os.listdir(sub_dir):
#     if d not in count_dic1:
#         shutil.rmtree(f'{sub_dir}{d}/')
#         print(d)
#
# len(os.listdir(sub_dir))

log_dir = '/home/ramaudruz/slurm_logs/genial/sbatch_error/'

log_files = os.listdir(log_dir)

#
# for i, f in enumerate(log_files):
#     if i % 1000 == 0:
#         print(i)
#     try:
#         os.remove(f'{log_dir}{f}')
#     except:
#         print(i)
#



count_dic2 = {k: v for k, v in count_dic.items() if v>= 8}

count_dic3 = {k: v for k, v in count_dic.items() if v < 8}


gen_dir = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'

#
# for d in os.listdir(gen_dir):
#     if d not in count_dic2:
#         try:
#             shutil.rmtree(gen_dir + d)
#         except:
#             print(f'Fail {d}')
#


count_dic_1 = {d: c for d, c in count_dic.items() if c == 1}
count_dic_2 = {d: c for d, c in count_dic.items() if c == 2 or c == 3}

count_1_df = pd.DataFrame([{'k': k, 'c': 1} for k in count_dic_1])
count_1_df.to_csv('/scratch/ramaudruz/misc/count_1_df_260105.csv', index=False)

# count_1_df = pd.DataFrame([{'k': k, 'c': 1} for k in count_dic_1])
# count_1_df.to_csv('/scratch/ramaudruz/misc/count_1_df_251224.csv', index=False)

# count_23_df = pd.DataFrame([{'k': k, 'c': 1} for k in count_dic_2])
# count_23_df.to_csv('/scratch/ramaudruz/misc/count_23_df_251224.csv', index=False)

to_remove = [
    d for d in os.listdir(data_dir)
    if not os.path.isfile(f'{data_dir}{d}/flowy_data_record.parquet')
    or pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0] != 12
]

count_dic_check = {
    d: (
        pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].shape[0]
        /
        pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0]
    )

    for d in suc_list}

# for d in to_remove:
#     shutil.rmtree(f'{data_dir}{d}')

new_to_delete = [d for d in os.listdir(data_dir) if (len(os.listdir(f'{data_dir}{d}')) == 0) or not os.path.isfile(f'{data_dir}{d}/flowy_data_record.parquet')]
# for d in new_to_delete:
#     shutil.rmtree(f'{data_dir}{d}')


df = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')

import os
home_dir = '/home/ramaudruz/proj/GENIAL/'

home_dir = '/scratch/ramaudruz/docker/data-root/containers/'

dir_list2 = os.listdir(home_dir)


# dir_list2 = [f for f in os.listdir(home_dir) if f.startswith('pymp')]

print(len(dir_list2))

# for i, d in enumerate(dir_list2):
#     try:
#         shutil.rmtree(f'{home_dir}{d}')
#     except:
#         pass
#     if i % 10 == 0:
#         print(i)


import shutil
from concurrent.futures import ThreadPoolExecutor

# def delete_many(batch):
#     for d in batch:
#         try:
#             shutil.rmtree(f'{home_dir}{d}')
#         except:
#             pass
#
# batch_size = 50
# batches = [dir_list2[i:i+batch_size] for i in range(0, len(dir_list2), batch_size)]
#
# with ThreadPoolExecutor(max_workers=128) as pool:
#     pool.map(delete_many, batches)




data_dir = '/scratch/mbouvier/proj/output_dgfe/output/multiplier_4bi_8bo_permuti_flowy/flowy_run_12chains_3000steps_proto_iter15/synth_out/'


res_list = os.listdir(data_dir)

suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()

print(suc_list)

# df = pd.read_parquet('/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/res_00000000000000/flowy_data_record.parquet')


count_list = [pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0] for d in suc_list]


d = suc_list[0]


#
# ##############################
#
#
#
# import pandas as pd
# import json
# import os
#
# data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/uniform_small/synth_out/'
#
# all_times = []
# for d in os.listdir(data_dir):
#     with open(f"{data_dir}{d}/data_record.json", "r") as f:
#         data = json.load(f)
#         time = data['runtime_metrics_extraction']
#         all_times.append(time['value'])
#
#
# print(f'time: {sum(all_times) / len(all_times)}')
#
# df_dict = {}
# for d in os.listdir(data_dir):
#     df_dict[d] = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
#
#
# ##############################
#
#
#
# import pandas as pd
# import json
# import os
#
# data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/uniform_small_SHORT/synth_out/'
#
# all_times = []
# for d in os.listdir(data_dir):
#     with open(f"{data_dir}{d}/data_record.json", "r") as f:
#         data = json.load(f)
#         time = data['runtime_metrics_extraction']
#         all_times.append(time['value'])
#
#
# print(f'time: {sum(all_times) / len(all_times)}')
#
# df_dict2 = {}
# for d in os.listdir(data_dir):
#     df_dict2[d] = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
#
# ########################
#
#
# import pandas as pd
# import json
# import os
#
# data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/uniform_small_SHORT/synth_out/'
#
# all_times = []
# for d in os.listdir(data_dir):
#     with open(f"{data_dir}{d}/data_record.json", "r") as f:
#         data = json.load(f)
#         time = data['runtime_metrics_extraction']
#         all_times.append(time['value'])
#
#
# print(f'time: {sum(all_times) / len(all_times)}')
#
# df_dict2 = {}
# for d in os.listdir(data_dir):
#     df_dict2[d] = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
#
#
#
#
#
# ########################
#
#
# import pandas as pd
# import json
# import os
#
# data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/uniform_small_SHORT2/synth_out/'
#
# all_times = []
# for d in os.listdir(data_dir):
#     with open(f"{data_dir}{d}/data_record.json", "r") as f:
#         data = json.load(f)
#         time = data['runtime_metrics_extraction']
#         all_times.append(time['value'])
#
#
# print(f'time: {sum(all_times) / len(all_times)}')
#
# df_dict3 = {}
# for d in os.listdir(data_dir):
#     df_dict3[d] = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
#



import pandas as pd
import os
import shutil

data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'


suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()

# print(suc_list)


# df = pd.read_parquet('/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/res_00000000000000/flowy_data_record.parquet')


# count_list = [pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0] for d in suc_list]

count_dic = {d: pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0] for d in suc_list}


len(os.listdir('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out_cache_251224'))




ref_df = pd.read_csv('/home/ramaudruz/data_dir/count_1_df_260105.csv')

completed_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_cache_260105'


done = os.listdir(completed_dir)

ref_df2 = ref_df[~ref_df['k'].isin(done)].reset_index(drop=True)

gen_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'

gen_dir_all = os.listdir(gen_dir)

to_keep = [d for d in gen_dir_all if d in ref_df2['k'].astype(str).tolist()]

to_keep_set = set(to_keep)

to_delete = [d for d in gen_dir_all if d not in to_keep_set]

for d in to_delete:
    shutil.rmtree(f'{gen_dir}/{d}')



min_mean_dic = {}

for i, d in enumerate(suc_list):
    if i % 100 == 0:
        print(i)
    min_mean_dic[d] = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet').groupby('run_identifier')['nb_transistors'].min().mean()


df_analysis = pd.DataFrame([{'d': k, 'min_mean': v} for k, v in min_mean_dic.items()]).sort_values('min_mean').reset_index(drop=True)



pd.read_parquet(f'{data_dir}{"res_00000000021816"}/flowy_data_record.parquet')['run_identifier'].unique().shape


f2 = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/res_00000000021816/hdl/mydesign_comb.v.bz2'



import bz2

path = "/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_3007_n_flips/generation_out/res_00000000000009/hdl/mydesign_comb.v.bz2"

with bz2.open(f2, "rt") as f:   # "rt" = read text
    content = f.read()



################################


count_dic2 = {k: v for k, v in count_dic.items() if v ==6}



df_dic = {}

for d in count_dic2:
    df_dic[d] = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet').groupby('run_identifier')['nb_transistors'].min().reset_index()

import numpy as np
data_list = []

for df in df_dic.values():
    data_list.append({
        'val': df['nb_transistors'].mean(),
        'std': df['nb_transistors'].std(),
        'log_val': pd.Series(np.log(df['nb_transistors'])).mean(),
        'log_std': pd.Series(np.log(df['nb_transistors'])).std(),
    })

df_std = pd.DataFrame(data_list).sort_values('val').reset_index(drop=True)


df_std.plot.scatter('val', 'std')

plt.show()


df_std.plot.scatter('log_val', 'log_std')

plt.show()



data_list = []

for df in df_dic.values():
    rolling = df['nb_transistors'].rolling(200).mean()

    break

    data_list.append({
        'val': df['nb_transistors'].mean(),
        'std': df['nb_transistors'].std(),
        'log_val': pd.Series(np.log(df['nb_transistors'])).mean(),
        'log_std': pd.Series(np.log(df['nb_transistors'])).std(),
        'rolling_5000': rolling.iloc[-1000],
        'rolling_6000': rolling.iloc[-1],
    })



##################

data_list = []
df_dic = {}

for d in count_dic2:
    df = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
    df_gr = df.groupby('run_identifier')['nb_transistors'].min().reset_index()

    rolling = df.groupby('step')['nb_transistors'].mean().reset_index()['nb_transistors'].rolling(200).mean()

    data_list.append({
        'd': d,
        'val': df_gr['nb_transistors'].mean(),
        'std': df_gr['nb_transistors'].std(),
        'log_val': pd.Series(np.log(df_gr['nb_transistors'])).mean(),
        'log_std': pd.Series(np.log(df_gr['nb_transistors'])).std(),
        'rolling_5000': rolling.iloc[-1000],
        'rolling_6000': rolling.iloc[-1],
    })



df_new = pd.DataFrame(data_list).sort_values('val').reset_index(drop=True)


res_00000000013837 = pd.read_parquet(f'{data_dir}res_00000000013837/flowy_data_record.parquet')



res_00000000013837_conc = pd.concat([
    res_00000000013837[res_00000000013837['run_identifier'] == r].reset_index(drop=True)['nb_transistors'].rolling(100).min()
    for r in res_00000000013837['run_identifier'].unique()
], axis=1)


ax = res_00000000013837_conc.plot()
ax.set_ylim(400, 800)
plt.show()


import bz2

path = "/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/res_00000000013837/hdl/mydesign_comb.v.bz2"

with bz2.open(path, "rt") as f:   # "rt" = read text
    content = f.read()


data_list = []

count_dic2 = {k: v for k, v in count_dic.items() if v >=3}
for d in count_dic2:
    df = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
    mean_min = df.groupby('run_identifier')['nb_transistors'].min().mean()
    data_list.append({
        'd': d,
        'mean_min': mean_min,
        'all_min': df['nb_transistors'].min(),
    })

anal_df = pd.DataFrame(data_list).sort_values('mean_min').reset_index(drop=True)




import bz2

path = "/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/res_00000000002634/hdl/mydesign_comb.v.bz2"

with bz2.open(path, "rt") as f:   # "rt" = read text
    content = f.read()






import pandas as pd
import os
import shutil


df_done = pd.read_parquet('/scratch/ramaudruz/misc/done_encs/synth_analysis.db.pqt')

df_done['design_number2'] = 'res_' + df_done['design_number']


gen_dir = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'

len(os.listdir(gen_dir))


to_delete = []

for d in os.listdir(gen_dir):
    if d in df_done['design_number2'].tolist():
        to_delete.append(d)
        shutil.rmtree(gen_dir + d)

len(os.listdir(gen_dir))





