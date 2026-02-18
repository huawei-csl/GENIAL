import os
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from genial.utils.utils import extract_cont_str, convert_cont_str_to_np

data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'
gen_data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'


data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_ALL_260216/'


data_dir2 = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_ALL_260216/'
data_dir1 = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_cache_88_260216/'
#
# len(os.listdir(data_dir1))
#
# for d in os.listdir(data_dir1):
#     if len(os.listdir(f'{data_dir1}{d}')):
#         if not os.path.exists(f'{data_dir2}{d}'):
#             shutil.copytree(f'{data_dir1}{d}', f'{data_dir2}{d}')
#         elif os.path.exists(f'{data_dir2}{d}/flowy_data_record.parquet'):
#             df_temp = pd.read_parquet(f'{data_dir1}{d}/flowy_data_record.parquet')
#             df_temp2 = pd.read_parquet(f'{data_dir2}{d}/flowy_data_record.parquet')
#             comb = pd.concat([df_temp, df_temp2], ignore_index=True)
#             comb.to_parquet(f'{data_dir2}{d}/flowy_data_record.parquet', index=False)


anal_df = pd.read_parquet(f'/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/analysis_out/synth_analysis.db.pqt')

completed_d = set(os.listdir(data_dir))

for d in os.listdir(gen_data_dir):
    if d not in completed_d:
        shutil.rmtree(gen_data_dir+d)


len(os.listdir(gen_data_dir))

len(
    set(os.listdir(data_dir1)) &
    set(os.listdir(data_dir2))
)

suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()

count_dic = {}
count_dic2 = {}

data_list = []

for i, d in enumerate(suc_list):

    df = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')

    if i % 100 == 0:
        print(i)
        print(df['run_identifier'].value_counts())

    cond = df['runtime_full_mockturtle_step'] > 1000

    com_run = df[cond]['run_identifier'].unique().tolist()

    data_list += [{'d': d, 'run_identifier': r} for r in com_run]

count_df = pd.DataFrame(data_list)

sum_count_df = count_df['d'].value_counts().reset_index()







# d_to_keep = sum_count_df[sum_count_df['count']>= 5]['d'].unique().tolist()
#
# for d in os.listdir(data_dir):
#     if d not in d_to_keep:
#         shutil.rmtree(f'{data_dir}/{d}')
#
#
#
#
# for c in common_list:
#     df1 = pd.read_parquet(f'{data_dir1}{c}/flowy_data_record.parquet')
#     df2 = pd.read_parquet(f'{data_dir2}{c}/flowy_data_record.parquet')
#     break



sum_count_df.to_csv('/home/ramaudruz/data_dir/misc/run_count_analysis/synth_out_cache_260212_df.csv', index=False)


sum_count_df10 = pd.read_csv('/home/ramaudruz/data_dir/misc/run_count_analysis/synth_out_cache_260212_df.csv')



gen_data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'



data_dir2 = '/scratch/ramaudruz/synth_out_260211/'
gen_data_dir2 = '/scratch/ramaudruz/generation_out_260211/'


set(os.listdir(gen_data_dir)) & set(os.listdir(data_dir2))

for d in os.listdir(data_dir2):
    shutil.copytree(data_dir2 + d, data_dir + d, dirs_exist_ok=True)
    if d in os.listdir(gen_data_dir2):
        print(d)
        shutil.copytree(gen_data_dir2 + d, gen_data_dir + d, dirs_exist_ok=True)

    if d in os.listdir(gen_data_dir):
        print('s')
    else:
        print('f')





suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) >= 0]
suc_list.sort()

all_to_keep = []
for k in suc_list:
    df = pd.read_parquet(f'{data_dir}{k}/flowy_data_record.parquet')
    to_keep = df[df['runtime_full_mockturtle_step'] > 1000]['run_identifier'].unique().tolist()

    if to_keep:
        df = df[df['run_identifier'].isin(to_keep)].reset_index(drop=True)
        df.to_parquet(f'{data_dir}{k}/flowy_data_record.parquet', index=False)
    else:
        shutil.rmtree(f'{data_dir}{k}')





#
in_dir = '/home/ramaudruz/data_dir/high_effort_raw_data/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'

for d in os.listdir(in_dir):
    if os.path.isfile(f'{in_dir}{d}/flowy_data_record.parquet'):
        df1 = pd.read_parquet(f'{in_dir}{d}/flowy_data_record.parquet')

        if not os.path.isfile(f'{data_dir}{d}/flowy_data_record.parquet'):
            df1.to_parquet(f'{data_dir}{d}/flowy_data_record.parquet', index=False)
        else:
            df2 = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
            df_conc = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)
            df_conc = df_conc.drop_duplicates(subset=['run_identifier', 'step'], keep='first').sort_values(['run_identifier', 'step']).reset_index(drop=True)
            df_conc.to_parquet(f'{data_dir}{d}/flowy_data_record.parquet', index=False)

count_dic2 = {k: v for k, v in count_dic.items() if v ==7}
#
#
# df_check =  {k: pd.read_parquet(f'{data_dir}{k}/flowy_data_record.parquet').groupby('run_identifier')['nb_transistors'].min() for k, v in count_dic2.items()}


import bz2

path = "/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/res_00000000004737/hdl/mydesign_comb.v.bz2"
path = "/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/res_00000000017326/hdl/mydesign_comb.v.bz2"

with bz2.open(path, "rt") as f:   # "rt" = read text
    content = f.read()




count_dic2 = {k: v for k, v in count_dic.items() if v >= 5}

all_to_keep = []
for k in count_dic2:
    df = pd.read_parquet(f'{data_dir}{k}/flowy_data_record.parquet')
    to_keep = df[df['runtime_full_mockturtle_step'] > 1000]['run_identifier'].unique().tolist()

    if to_keep:
        df = df[df['run_identifier'].isin(to_keep)].reset_index(drop=True)
        df.to_parquet(f'{data_dir}{k}/flowy_data_record.parquet', index=False)
    else:
        shutil.rmtree(f'{data_dir}{k}')



df_check =  {k: pd.read_parquet(f'{data_dir}{k}/flowy_data_record.parquet').groupby('run_identifier')['nb_transistors'].min().mean() for k, v in count_dic2.items()}

#
# data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'

data_list = []

count_dic2 = {k: v for k, v in count_dic.items() if v >=5}
for d in count_dic2:
    df = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
    mean_min = df.groupby('run_identifier')['nb_transistors'].min().mean()
    mean_std = df.groupby('run_identifier')['nb_transistors'].min().std()
    data_list.append({
        'd': d,
        'mean_min': mean_min,
        'mean_std': mean_std,
        'all_min': df['nb_transistors'].min(),
    })

anal_df2 = pd.DataFrame(data_list).sort_values('d').reset_index(drop=True)

mean_min_dic = dict(zip(anal_df2['d'], anal_df2['mean_min']))

to_keep = set(count_dic2.keys())




preds = torch.load('/home/ramaudruz/data_dir/misc/pred_verif_260216/val_preds.pt')
encods = torch.load('/home/ramaudruz/data_dir/misc/pred_verif_260216/val_encodings.pt')


dic_list = []

for enc_count, e in enumerate(encods):
    temp_dic = {}
    for i, f in enumerate(e):
        temp_dic[i - 8] = ''.join([str(int(g)) for g in f])
    dic_list.append({'dic_str': str(temp_dic), 'pred': preds[enc_count]})

pred_df = pd.DataFrame(dic_list)

sum_df = pd.read_csv('/home/ramaudruz/data_dir/misc/pred_verif_260216/total_df.csv')

sum_df['encodings_input_gener'].iloc[0]

trans_dict = dict(zip(sum_df['encodings_input_gener'], sum_df['nb_transistors']))


pred_df['nb_transistors'] = pred_df['dic_str'].map(trans_dict)

###################################################

# Derive a boolean matrix representation of the encoding sequence
encodings_input_np = (
    pred_df['dic_str']
    .map(lambda x: extract_cont_str(x))
    .map(lambda x: convert_cont_str_to_np(x).reshape((16, 4)).T.astype(np.bool_))
)
# Derive the flipped representation of the encoding sequence
encodings_input_np_flipped = encodings_input_np.map(lambda x: ~x)

# Sort the columns and obtain by representation for unflipped and flipped version
encodings_input_np_sorted = encodings_input_np.map(lambda x: (x[np.lexsort(x.T[::-1])]).tobytes())
encodings_input_np_flipped_sorted = encodings_input_np_flipped.map(
    lambda x: (x[np.lexsort(x.T[::-1])]).tobytes()
)

# Obtain unique representation for the flipped and unflipped version
pred_df["encodings_input_group_id"] = np.minimum(encodings_input_np_sorted, encodings_input_np_flipped_sorted)

###########################################################

# Derive a boolean matrix representation of the encoding sequence
encodings_input_np = (
    sum_df['encodings_input_gener']
    .map(lambda x: extract_cont_str(x))
    .map(lambda x: convert_cont_str_to_np(x).reshape((16, 4)).T.astype(np.bool_))
)
# Derive the flipped representation of the encoding sequence
encodings_input_np_flipped = encodings_input_np.map(lambda x: ~x)

# Sort the columns and obtain by representation for unflipped and flipped version
encodings_input_np_sorted = encodings_input_np.map(lambda x: (x[np.lexsort(x.T[::-1])]).tobytes())
encodings_input_np_flipped_sorted = encodings_input_np_flipped.map(
    lambda x: (x[np.lexsort(x.T[::-1])]).tobytes()
)

# Obtain unique representation for the flipped and unflipped version
sum_df["encodings_input_group_id"] = np.minimum(encodings_input_np_sorted, encodings_input_np_flipped_sorted)

sum_df2 = sum_df.groupby("encodings_input_group_id")[['nb_transistors', 'scores']].mean().reset_index()


trans_dict = dict(zip(sum_df2['encodings_input_group_id'], sum_df2['nb_transistors']))

pred_df["nb_transistors"] = pred_df["encodings_input_group_id"].map(trans_dict)


scores_dict = dict(zip(sum_df2['encodings_input_group_id'], sum_df2['scores']))

pred_df["scores"] = pred_df["encodings_input_group_id"].map(scores_dict)

pred_df["pred2"] = pred_df["pred"].map(lambda x: float(x[0]))

plt.figure(figsize=(8, 6))
plt.scatter(pred_df["pred2"], pred_df["scores"])
plt.show()


