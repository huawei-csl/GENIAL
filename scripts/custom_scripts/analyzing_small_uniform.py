
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr

# Data dir
data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/uniform_small/synth_out/'

# Design list
des_list = os.listdir(data_dir)
des_list.sort()

# Df dict
df_dict = {d: pd.read_parquet(data_dir + f'{d}/flowy_data_record.parquet') for d in des_list}

# Df_dict short
df_dict_short = {}
for d, df in df_dict.items():
    df_dict_short[d] = df[df['step'] < 3000].reset_index(drop=True)

# Min dict
min_dict = {d: df['nb_transistors'].min() for d, df in df_dict.items()}

# data collection dict
data_list = []

for d, df in df_dict_short.items():

    print(d)

    df_gr = df.groupby('run_identifier')['nb_transistors'].min().reset_index()
    df_gr = df_gr.sort_values('nb_transistors').reset_index(drop=True)

    run_list = df['run_identifier'].unique().tolist()

    for count in range(50000):
        selected_runs = np.random.choice(run_list, size=12, replace=False)

        df_temp = df_gr[df_gr['run_identifier'].isin(selected_runs)]

        data_list.append({
            'run_identifier': d,
            'count': count,
            'min': df_temp['nb_transistors'].iloc[0],
            'best_2': df_temp['nb_transistors'].iloc[:2].mean(),
            'best_3': df_temp['nb_transistors'].iloc[:3].mean(),
            'best_4': df_temp['nb_transistors'].iloc[:4].mean(),
            'best_5': df_temp['nb_transistors'].iloc[:5].mean(),
            'mean': df_temp['nb_transistors'].mean(),
        })



min_df = pd.DataFrame([{'run_identifier': d, 'min_trans': m} for d, m in min_dict.items()])
min_df = min_df.sort_values('run_identifier').reset_index(drop=True)

data_col_df = pd.DataFrame(data_list)
data_col_df = data_col_df.sort_values(['count', 'run_identifier']).reset_index(drop=True)

new_df_list = []
for c in data_col_df['count'].unique():
    df_temp = data_col_df[data_col_df['count'] == c]
    for col in ('min', 'best_2', 'best_3', 'best_4', 'best_5', 'mean'):
        rho_s, _ = spearmanr(df_temp[col].values, min_df['min_trans'].values)
        rho_p, _ = pearsonr(df_temp[col].values, min_df['min_trans'].values)
        new_df_list.append({
            'count': c,
            'col': col,
            'rho_s': rho_s,
            'rho_p': rho_p,
        })


stats_df = pd.DataFrame(new_df_list)

for col in ('min', 'best_2', 'best_3', 'best_4', 'best_5', 'mean'):

    mean_spearman = stats_df[stats_df['col']==col]['rho_s'].mean()
    std_spearman = stats_df[stats_df['col']==col]['rho_s'].std()

    mean_pearson  = stats_df[stats_df['col']==col]['rho_p'].mean()
    std_pearson  = stats_df[stats_df['col']==col]['rho_p'].std()

    print(col)
    print(f"Spearman: {mean_spearman:.3f} ± {std_spearman:.3f}")
    print(f"Pearson: {mean_pearson:.3f} ± {std_pearson:.3f}")
    print('\n')
