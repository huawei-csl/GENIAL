
import os
import h5py
import tarfile
import io
import pandas as pd
import numpy as np
import polars as pl
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def read_and_tag(round_num, mig_dir):
    parquet_file = mig_dir + f"mig_output_round_{round_num}.parquet"
    if not os.path.isfile(parquet_file):
        return None  # or raise an exception if you want to stop
    df = pd.read_parquet(parquet_file)
    df.insert(0, "round", round_num)
    return df

# Synth out dir
enc_dir = "/mnt/nvme/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/gnn_data_collection/synth_out/"

df_list = []

# Loop through encodings
for dir_count, d in enumerate(os.listdir(enc_dir)):
    # Print log
    if dir_count % 10 == 0:
        print(dir_count)
    # Retrieve run dirs
    run_dirs = [rd for rd in os.listdir(enc_dir + d) if rd.startswith("run")]
    # Loop through run dirs
    for run_count, run_dir in enumerate(run_dirs):
        run_dir_full = f"{enc_dir}{d}/{run_dir}/"
        flowy_parquet = run_dir_full + "flowy_record.parquet"
        if os.path.isfile(flowy_parquet):
            df = pd.read_parquet(flowy_parquet)
            df['enc_id'] = d
            # df['run_id'] = run_dir
            del df['recipe_round']
            del df['score']

            df['gates'] = df['gates'].astype('int16')
            df['depth'] = df['depth'].astype('int16')

            df_list.append(df)


df_all = pd.concat(df_list, ignore_index=True)

# Cute gates
df_all['gates_bin'] = pd.cut(df_all['gates'], bins=100)

# Encoding group
df_all['group_id'] = df_all['gates_bin'].astype(str) + '_' + df_all['depth'].astype(str) + '_' + df_all['enc_id'].astype(str)

# Gate depth group
df_all['group_id2'] = df_all['gates_bin'].astype(str) + '_' + df_all['depth'].astype(str)

# Gate depth group value counts
gate_depth_vc = df_all['group_id2'].value_counts()

# First subset
df_all_sub1 = df_all[df_all['group_id2'].isin(gate_depth_vc[gate_depth_vc < 500].index.tolist())]

# Encoding count dict
df_all_sub1_vc = df_all_sub1['enc_id'].value_counts().reset_index()
enc_count = dict(zip(df_all_sub1_vc['enc_id'], df_all_sub1_vc['count']))
missing_from_first = list(set(df_all['enc_id'].tolist()) - set(df_all_sub1['enc_id'].tolist()))
for e in missing_from_first:
    enc_count[e] = 0

min_e_count = min([v for v in enc_count.values()])
to_add = 500 - min_e_count

min_e_map = {k: v for k, v in enc_count.items() if v == min_e_count}

# Second subset
df_all_sub2 = df_all[df_all['group_id2'].isin(gate_depth_vc[gate_depth_vc >= 500].index.tolist())]

new_sample_df = None
count_dic = None

for e in min_e_map:
    print(e)

    # Select encoding specific rows
    df_all_sub2_e = df_all_sub2[df_all_sub2['enc_id'] == e]

    # Shuffle the df
    df_all_sub2_e = df_all_sub2_e.iloc[np.random.permutation(len(df_all_sub2_e))]

    # Drop_duplicates
    df_all_sub2_e_u = df_all_sub2_e.drop_duplicates('group_id').reset_index(drop=True)

    if df_all_sub2_e_u.shape[0] <= to_add:
        if new_sample_df is None:
            new_sample_df = df_all_sub2_e_u.reset_index(drop=True)
        else:
            new_sample_df = pd.concat([new_sample_df, df_all_sub2_e_u], ignore_index=True)
    else:
        if new_sample_df is None:
            new_sample_df = df_all_sub2_e_u.iloc[:to_add].reset_index(drop=True)
        else:
            df_all_sub2_e['temp_count'] = 0
            temp_vc = new_sample_df['group_id2'].value_counts().reset_index()
            value_count_map = dict(zip(temp_vc['group_id2'], temp_vc['count']))
            cond = df_all_sub2_e_u['group_id2'].isin(value_count_map)
            df_all_sub2_e_u.loc[cond, 'temp_count'] = df_all_sub2_e_u.loc[cond, 'group_id2'].map(value_count_map)
            df_all_sub2_e_u = df_all_sub2_e_u.sort_values('temp_count').reset_index(drop=True)
            df_all_sub2_e_u = df_all_sub2_e_u.drop(columns='temp_count')
            new_sample_df = pd.concat([new_sample_df, df_all_sub2_e_u.iloc[:to_add]], ignore_index=True)




df_all_sub2b = df_all_sub2.drop_duplicates('group_id')

df_comb1 = pd.concat([df_all_sub1, df_all_sub2b], ignore_index=True)





# # Encoding count dict with less than 100 samples
# less_than_100 = [k for k, v in enc_count.items() if v < 100]
#
# # Remaining bins and not so selected encodings
# cond1 = df_all['group_id2'].isin(gate_depth_vc[gate_depth_vc >= 500].index.tolist())
# cond2 = df_all['enc_id'].isin(less_than_100)
# df_all_temp_sub = df_all[cond1 & cond2].drop_duplicates('group_id')
#
# # Combine
# df_comb1 = pd.concat([df_all_sub1, df_all_temp_sub], ignore_index=True)
#
# # Encoding count dict 2
# df_comb1_vc = df_comb1['enc_id'].value_counts().reset_index()
# enc_count2 = dict(zip(df_comb1_vc['enc_id'], df_comb1_vc['count']))
#
#
# list(set(df_all['group_id2'].tolist()) - set(df_comb1['group_id2'].tolist()))
#
#
#
# # Remaining gate depth groups
# gate_depth_remaining = gate_depth_vc[gate_depth_vc >= 500].index.tolist()
#
# df_all_temp_sub = df_all[df_all['group_id2'].isin(gate_depth_vc[gate_depth_vc >= 500].index.tolist())]
#
# df_all_temp_sub2 = df_all_temp_sub.drop_duplicates('group_id')
