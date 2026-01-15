


import os
import shutil
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# synth_out1 = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_cache_260105/'
#
# synth_out2 = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_cache_260106/'
#
#
#
# synth_list1 = os.listdir(synth_out1)
#
# synth_list2 = os.listdir(synth_out2)


# act_synth = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_act/'
#
# act_synth_list = os.listdir(act_synth)

# synth_1_df_dic = {}
# for d in synth_list1:
#     synth_1_df_dic[d] = pd.read_parquet(synth_out1 + d + '/flowy_data_record.parquet')
#
# synth_2_df_dic = {}
# for d in synth_list2:
#     synth_2_df_dic[d] = pd.read_parquet(synth_out2 + d + '/flowy_data_record.parquet')
#
#
# synth_comb = {
#     **synth_1_df_dic,
#     **synth_2_df_dic,
# }
#
# partial_df_dic = {}
# for d in synth_comb:
#     partial_df_dic[d] = pd.read_parquet(act_synth + d + '/flowy_data_record.parquet')
#
#
# full_df_dic = {}
# for d in partial_df_dic:
#     full_df_dic[d] = pd.concat([
#         partial_df_dic[d],
#         synth_comb[d]
#     ], ignore_index=True)
#
# for d in full_df_dic:
#     full_df_dic[d].to_parquet(act_synth + d + '/flowy_data_record.parquet', index=False)


# synth_comb_gr = {}
# for d, df in synth_comb.items():
#     synth_comb_gr[d] = df.groupby('run_identifier')['nb_transistors'].min().reset_index()
#
# partial_df_dic_gr = {}
# for d, df in partial_df_dic.items():
#     partial_df_dic_gr[d] = df.groupby('run_identifier')['nb_transistors'].min().reset_index()
#
# for d in synth_comb_gr:
#     min_val = synth_comb_gr[d]['nb_transistors'].min()
#     max_val = synth_comb_gr[d]['nb_transistors'].max()
#     partial_val = partial_df_dic_gr[d]['nb_transistors'].max()
#     print(f"min: {min_val}, max: {max_val}, partial: {partial_val}")
#
#
# for d in synth_comb_gr:
#     mean_val = synth_comb_gr[d]['nb_transistors'].mean()
#     partial_val = partial_df_dic_gr[d]['nb_transistors'].max()
#     # print(f"diff: {(mean_val - partial_val) / mean_val}")
#
#     if np.abs((mean_val - partial_val) / mean_val) > 0.2:
#         min_val = synth_comb_gr[d]['nb_transistors'].min()
#         max_val = synth_comb_gr[d]['nb_transistors'].max()
#         partial_val = partial_df_dic_gr[d]['nb_transistors'].max()
#         print(f"min: {min_val}, max: {max_val}, partial: {partial_val}")
#
#



# to_delete = []
#
# for d in act_synth_list:
#     if not os.path.isfile(act_synth + d + '/flowy_data_record.parquet'):
#         to_delete.append(d)
#
# len(to_delete)
#
# len(act_synth_list)
#
# for d in to_delete:
#     shutil.rmtree(act_synth + d)
#
#



# act_synth = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_act/'
#
# act_synth_list = os.listdir(act_synth)
#
#
# count_dic = {}
#
# for i, d in enumerate(act_synth_list):
#     if i % 100 == 0:
#         print(i)
#     count_dic[d] = pd.read_parquet(f'{act_synth}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0]
#
#
# pd.Series(count_dic.values()).value_counts()
#
#
# min_track = 10000
#
# for i, d in enumerate(act_synth_list):
#     curr_min = pd.read_parquet(f'{act_synth}{d}/flowy_data_record.parquet')['nb_transistors'].min()
#     if curr_min < min_track:
#         min_track = curr_min
#
#
# act_synth_new = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_act_cons/'
# count_dic = {}
#
# for i, d in enumerate(os.listdir(act_synth_new)):
#     if i % 100 == 0:
#         print(i)
#     count_dic[d] = pd.read_parquet(f'{act_synth_new}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0]
#
#
#
# synth_out = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'
# gen_out = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'
#
# len(os.listdir(gen_out))
#
#
# synth_set = set(os.listdir(synth_out))
#
# for d in os.listdir(gen_out):
#     if d not in synth_set:
#         shutil.rmtree(f'{gen_out}{d}')


# act_synth_new = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_act_cons/'
# count_dic = {}
#
# for i, d in enumerate(os.listdir(act_synth_new)):
#     if i % 100 == 0:
#         print(i)
#     count_dic[d] = int(pd.read_parquet(f'{act_synth_new}{d}/flowy_data_record.parquet').groupby('run_identifier')['nb_transistors'].min().mean())
#
# count_dic2 = {k[4:]: v for k, v in count_dic.items()}



# df_check = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/analysis_out/synth_analysis.db.pqt')
#
#
# df_swact = pd.read_csv('/home/ramaudruz/data_dir/misc/swact_data_with_encoding.csv')
#
# df_swact_gr = df_swact.groupby('encodings_input')['min_val'].min().reset_index()
#
# swact_dict = dict(zip(df_swact_gr['encodings_input'], df_swact_gr['min_val']))
#
#
# df_check['swact'] = df_check['encodings_input'].map(swact_dict)
#
# import matplotlib.pyplot as plt
#
# df_check.plot.scatter(x="nb_transistors", y="swact")
# plt.show()
#
#
#
# df_check['nb_transistors'].min()
#
#
# df_check1 = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme/synth_out/res_00000000000000/flowy_data_record.parquet')
#
# tc_mean = df_check1.groupby('run_identifier')['nb_transistors'].min().mean()
# tc_min = df_check1.groupby('run_identifier')['nb_transistors'].min().min()
#
# df_check2 = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme/synth_out/res_00000000000001/flowy_data_record.parquet')
#
# sme_mean = df_check2.groupby('run_identifier')['nb_transistors'].min().mean()
# sme_min = df_check2.groupby('run_identifier')['nb_transistors'].min().min()
#
# print(f'TC: mean {tc_mean}, min {tc_min}')
# print(f'SME: mean {sme_mean}, min {sme_min}')
#
#
# df_check1 = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme/synth_out_first_round/res_00000000000000/flowy_data_record.parquet')
#
# tc_mean = df_check1.groupby('run_identifier')['nb_transistors'].min().mean()
# tc_min = df_check1.groupby('run_identifier')['nb_transistors'].min().min()
#
# df_check2 = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme/synth_out_first_round/res_00000000000001/flowy_data_record.parquet')
#
# sme_mean = df_check2.groupby('run_identifier')['nb_transistors'].min().mean()
# sme_min = df_check2.groupby('run_identifier')['nb_transistors'].min().min()
#
# print(f'TC: mean {tc_mean}, min {tc_min}')
# print(f'SME: mean {sme_mean}, min {sme_min}')
#
#
# data_list = []
# synth_out = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'
#
# for d in os.listdir(synth_out):
#     df = pd.read_parquet(synth_out + d + '/flowy_data_record.parquet')
#     group_by_temp = df.groupby('run_identifier')
#     temp_mean = group_by_temp['nb_transistors'].min().mean()
#     temp_min = group_by_temp['nb_transistors'].min().min()
#
#     data_list.append({
#         'd': d,
#         'mean': temp_mean,
#         'min': temp_min,
#     })
#
#
# analysis_df = pd.DataFrame(data_list).sort_values('mean').reset_index(drop=True)
#
#
#
# import bz2
#
# path = "/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/res_00000000003007/hdl/mydesign_comb.v.bz2"
#
# with bz2.open(path, "rt") as f:   # "rt" = read text
#     content = f.read()
#
# d = 'res_00000000003007'
# df_3007 = pd.read_parquet(synth_out + d + '/flowy_data_record.parquet')
#
#
# df_3007_gr = df_3007.groupby('run_identifier')['nb_transistors'].min().reset_index()
#
#
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
#
# new_data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_n_3007/synth_out/'
#
# effort_df_dic = {}
#
# for d in os.listdir(new_data_dir):
#     df = pd.read_parquet(new_data_dir + d + '/flowy_data_record.parquet')
#     effort_df_dic[d] = df
#
#
#
# # for d, df in effort_df_dic.items():
# for d in ('res_00000000003007', 'res_00000000000001', 'res_00000000000000'):
#     df = effort_df_dic[d]
#     print(d)
#     print(f'Min: {df['nb_transistors'].min()}')
#     print(f'Mean of min: {df.groupby('run_identifier')['nb_transistors'].min().mean()}')
#     print(f'Min (at 3000): {df[df['step']<3000]['nb_transistors'].min()}')
#     print(f'Mean of min (at 3000): {df[df['step']<3000].groupby('run_identifier')['nb_transistors'].min().mean()}')
#     print(f'Min (at 6000): {df[df['step']<6000]['nb_transistors'].min()}')
#     print(f'Mean of min (at 6000): {df[df['step']<6000].groupby('run_identifier')['nb_transistors'].min().mean()}')
#     print(df.shape)
#
#
# effort_df_dic_mean_step = {}
#
# # for d, df in effort_df_dic.items():
# for d in ('res_00000000003007', 'res_00000000000001', 'res_00000000000000'):
#     df = effort_df_dic[d]
#     if d == 'res_00000000000000':
#         rename = 'TC'
#     elif d == 'res_00000000000001':
#         rename = 'SME'
#     else:
#         rename = '3007'
#     df['nb_transistors_cummin'] = df.groupby("run_identifier")['nb_transistors'].cummin()
#     effort_df_dic_mean_step[d] = df.groupby('step')['nb_transistors_cummin'].min().reset_index().rename(columns={'nb_transistors_cummin': rename}).set_index('step')
#
# df_conc = pd.concat(effort_df_dic_mean_step.values(), axis=1)
#
# ax = df_conc.plot()
# ax.set_ylim(360, 600)
# plt.show()
#
# effort_df_dic_mean_step = {}
#
# # for d, df in effort_df_dic.items():
# for d in ('res_00000000003007', 'res_00000000000001', 'res_00000000000000'):
#     df = effort_df_dic[d]
#     if d == 'res_00000000000000':
#         rename = 'TC'
#     elif d == 'res_00000000000001':
#         rename = 'SME'
#     else:
#         rename = '3007'
#     effort_df_dic_mean_step[d] = df.groupby('step')['nb_transistors'].min().reset_index().rename(columns={'nb_transistors': rename}).set_index('step')
#
# df_conc = pd.concat(effort_df_dic_mean_step.values(), axis=1)
#
# ax = df_conc.plot()
# ax.set_ylim(360, 600)
# plt.show()
#
#
# effort_df_dic_mean_step = {}
#
# # for d, df in effort_df_dic.items():
# for d in ('res_00000000003007', 'res_00000000000001', 'res_00000000000000'):
#     df = effort_df_dic[d]
#     if d == 'res_00000000000000':
#         rename = 'TC'
#     elif d == 'res_00000000000001':
#         rename = 'SME'
#     else:
#         rename = '3007'
#     df["nb_transistors"] = (
#         df.groupby("run_identifier")["nb_transistors"]
#         .rolling(window=30, min_periods=1)
#         .min()
#         .reset_index(level=0, drop=True)
#     )
#
#     effort_df_dic_mean_step[d] = df.groupby('step')['nb_transistors'].min().reset_index().rename(columns={'nb_transistors': rename}).set_index('step')
#
# df_conc = pd.concat(effort_df_dic_mean_step.values(), axis=1)
#
# ax = df_conc.plot()
# ax.set_ylim(360, 600)
# plt.show()
#
# effort_df_dic_mean_step = {}
#
# # for d, df in effort_df_dic.items():
# for s in (2999, 4999, 7999):
#     for d in ('res_00000000003007', 'res_00000000000001', 'res_00000000000000'):
#         df = effort_df_dic[d]
#         if d == 'res_00000000000000':
#             rename = 'TC'
#         elif d == 'res_00000000000001':
#             rename = 'SME'
#         else:
#             rename = '3007'
#         df['nb_transistors_cummin'] = df.groupby("run_identifier")['nb_transistors'].cummin()
#         df_temp = df[df['step'] == s]['nb_transistors_cummin']
#         ax = df_temp.hist(bins=12, figsize=(10, 4))
#         if s == 2999:
#             ax.set_xlim(370, 630)
#         if s == 4999:
#             ax.set_xlim(370, 620)
#         if s == 7999:
#             ax.set_xlim(370, 600)
#
#         ax.set_title(f'{rename} - step {s + 1}')
#
#         plt.show()
#
#
#
#
#
#
#
#
# import pandas as pd
# import numpy as np
# import os
# from scipy.stats import spearmanr, pearsonr
#
# # Data dir
# data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_n_3007/synth_out/'
#
# # Design list
# des_list = os.listdir(data_dir)
# des_list.sort()
#
# # Df dict
# df_dict = {d: pd.read_parquet(data_dir + f'{d}/flowy_data_record.parquet') for d in des_list}
#
# # Df_dict short
# df_dict_short = {}
# for d, df in df_dict.items():
#     df_dict_short[d] = df[df['step'] < 3000].reset_index(drop=True)
#
# # Min dict
# min_dict = {d: df['nb_transistors'].min() for d, df in df_dict.items()}
#
# # data collection dict
# data_list = []
#
# for d, df in df_dict_short.items():
#
#     print(d)
#
#     df_gr = df.groupby('run_identifier')['nb_transistors'].min().reset_index()
#     df_gr = df_gr.sort_values('nb_transistors').reset_index(drop=True)
#
#     run_list = df['run_identifier'].unique().tolist()
#
#     for count in range(5000):
#         selected_runs = np.random.choice(run_list, size=12, replace=False)
#
#         df_temp = df_gr[df_gr['run_identifier'].isin(selected_runs)]
#
#         mean = df_temp['nb_transistors'].mean()
#         std = df_temp['nb_transistors'].std()
#
#         data_list.append({
#             'run_identifier': d,
#             'count': count,
#             'min': df_temp['nb_transistors'].iloc[0],
#             'best_2': df_temp['nb_transistors'].iloc[:2].mean(),
#             'best_3': df_temp['nb_transistors'].iloc[:3].mean(),
#             'best_4': df_temp['nb_transistors'].iloc[:4].mean(),
#             'best_5': df_temp['nb_transistors'].iloc[:5].mean(),
#             'mean': mean,
#             'mean - 0.75 std': mean - 0.75 * std,
#             'mean - 1 std': mean - 1 * std,
#             'mean - 1.25 std': mean - 1.25 * std,
#             'mean - 1.5 std': mean - 1.5 * std,
#             'mean - 2 std': mean - 2 * std,
#         })
#
#
#
# min_df = pd.DataFrame([{'run_identifier': d, 'min_trans': m} for d, m in min_dict.items()])
# min_df = min_df.sort_values('run_identifier').reset_index(drop=True)
#
# data_col_df = pd.DataFrame(data_list)
# data_col_df = data_col_df.sort_values(['count', 'run_identifier']).reset_index(drop=True)
#
# new_df_list = []
# for c in data_col_df['count'].unique():
#     df_temp = data_col_df[data_col_df['count'] == c]
#     for col in ('min', 'best_2', 'best_3', 'best_4', 'best_5', 'mean', 'mean - 0.75 std', 'mean - 1 std', 'mean - 1.25 std', 'mean - 1.5 std', 'mean - 2 std'):
#         rho_s, _ = spearmanr(df_temp[col].values, min_df['min_trans'].values)
#         rho_p, _ = pearsonr(df_temp[col].values, min_df['min_trans'].values)
#         new_df_list.append({
#             'count': c,
#             'col': col,
#             'rho_s': rho_s,
#             'rho_p': rho_p,
#         })
#
#
# stats_df = pd.DataFrame(new_df_list)
#
# for col in ('min', 'best_2', 'best_3', 'best_4', 'best_5', 'mean', 'mean - 0.75 std', 'mean - 1 std', 'mean - 1.25 std', 'mean - 1.5 std', 'mean - 2 std'):
#
#     mean_spearman = stats_df[stats_df['col']==col]['rho_s'].mean()
#     std_spearman = stats_df[stats_df['col']==col]['rho_s'].std()
#
#     mean_pearson  = stats_df[stats_df['col']==col]['rho_p'].mean()
#     std_pearson  = stats_df[stats_df['col']==col]['rho_p'].std()
#
#     print(col)
#     print(f"Spearman: {mean_spearman:.3f} ± {std_spearman:.3f}")
#     print(f"Pearson: {mean_pearson:.3f} ± {std_pearson:.3f}")
#     print('\n')
#
#
# effort_df_dic_mean_step = {}
#
# # for d, df in effort_df_dic.items():
# for d in ('res_00000000003007', 'res_00000000000001', 'res_00000000000000'):
#     df = effort_df_dic[d]
#     if d == 'res_00000000000000':
#         rename = 'TC'
#     elif d == 'res_00000000000001':
#         rename = 'SME'
#     else:
#         rename = '3007'
#     effort_df_dic_mean_step[d] = df.groupby('step')['nb_transistors'].mean().reset_index().rename(columns={'nb_transistors': rename}).set_index('step')
#
# df_conc = pd.concat(effort_df_dic_mean_step.values(), axis=1)
#
# ax = df_conc.plot()
# ax.set_ylim(450, 800)
# plt.show()
#
# #################################################
#
#
#
# import pandas as pd
# import numpy as np
# import os
# from scipy.stats import spearmanr, pearsonr
#
# # Df dict
# df_dict = {d: pd.read_parquet(data_dir + f'{d}/flowy_data_record.parquet') for d in des_list}
#
# for df in df_dict.values():
#     df['nb_transistors_min_30'] = (
#         df.groupby("run_identifier")["nb_transistors"]
#         .rolling(window=30, min_periods=1)
#         .min()
#         .reset_index(level=0, drop=True)
#     )
#     df['nb_transistors_min_60'] = (
#         df.groupby("run_identifier")["nb_transistors"]
#         .rolling(window=60, min_periods=1)
#         .min()
#         .reset_index(level=0, drop=True)
#     )
#     df['nb_transistors_mean_30'] = (
#         df.groupby("run_identifier")["nb_transistors"]
#         .rolling(window=30, min_periods=1)
#         .mean()
#         .reset_index(level=0, drop=True)
#     )
#     df['nb_transistors_mean_60'] = (
#         df.groupby("run_identifier")["nb_transistors"]
#         .rolling(window=60, min_periods=1)
#         .mean()
#         .reset_index(level=0, drop=True)
#     )
#
# # # Min dict
# # min_dict = {d: df['nb_transistors'].min() for d, df in df_dict.items()}
#
# # data collection dict
# data_list = []
#
# for d, df in df_dict.items():
#
#
#     print(d)
#
#     # df_gr = df.groupby('run_identifier')['nb_transistors'].min().reset_index()
#     # df_gr = df_gr.sort_values('nb_transistors').reset_index(drop=True)
#
#     run_list = df['run_identifier'].unique().tolist()
#
#     # for count in range(5000):
#     #     selected_runs = np.random.choice(run_list, size=12, replace=False)
#     #
#     #     df_temp = df_gr[df_gr['run_identifier'].isin(selected_runs)]
#     #
#     #     mean = df_temp['nb_transistors'].mean()
#     #     std = df_temp['nb_transistors'].std()
#     #
#     #     data_list.append({
#     #         'run_identifier': d,
#     #         'count': count,
#     #         'min': df_temp['nb_transistors'].iloc[0],
#     #         'best_2': df_temp['nb_transistors'].iloc[:2].mean(),
#     #         'best_3': df_temp['nb_transistors'].iloc[:3].mean(),
#     #         'best_4': df_temp['nb_transistors'].iloc[:4].mean(),
#     #         'best_5': df_temp['nb_transistors'].iloc[:5].mean(),
#     #         'mean': mean,
#     #         'mean - 0.75 std': mean - 0.75 * std,
#     #         'mean - 1 std': mean - 1 * std,
#     #         'mean - 1.25 std': mean - 1.25 * std,
#     #         'mean - 1.5 std': mean - 1.5 * std,
#     #         'mean - 2 std': mean - 2 * std,
#     #     })
#
#     # df = df_dict['res_00000000003007']
#     df = df_dict['res_00000000000001']
#
#     df = df_dict['res_00000000000000']
#
#     run_list = df['run_identifier'].unique().tolist()
#
#     series_list = []
#
#     for _ in range(100):
#         selected_runs = np.random.choice(run_list, size=6, replace=False)
#         df_temp = df[(df['run_identifier'].isin(selected_runs)) & (df['step'] < 6000)]
#         series_list.append(df_temp.groupby('run_identifier')['nb_transistors'].min().mean())
#
#     pd.Series(series_list).std()
#     pd.Series(series_list).mean()
#
#
#
#     series_list = []
#     for _ in range(100):
#         selected_runs = np.random.choice(run_list, size=12, replace=False)
#         df_temp = df[(df['run_identifier'].isin(selected_runs)) & (df['step'] < 3000)]
#         series_list.append(df_temp.groupby('run_identifier')['nb_transistors'].min().mean())
#
#     pd.Series(series_list).std()
#     pd.Series(series_list).mean()
#
#
#
#         # df_temp_gr = df_temp.groupby('step')['nb_transistors'].mean().reset_index().rename(columns={'nb_transistors': rename}).set_index('step')
#         # series_list.append(df_temp_gr.rolling(window=1000, min_periods=1).mean())
#
#     df_conc = pd.concat(series_list, axis=1)
#
#     ax = df_conc.plot(legend=False)
#     ax.set_ylim(450, 650)
#     # ax.set_ylim(550, 750)
#     plt.show()
#
#
# value_list = []
# value_list2 = []
#
# df2 = df[df['step'] < 6000].reset_index(drop=True)
#
# df2_gr = df2.groupby('run_identifier')['nb_transistors'].min().reset_index().sort_values(by='nb_transistors').reset_index(drop=True)
#
# for i in range(5000):
#     value_list.append(
#         df2_gr[df2_gr['run_identifier'].isin(np.random.choice(df2_gr['run_identifier'].unique(), 6, replace=False))]['nb_transistors'].mean()
#     )
#     value_list2.append(
#         df2_gr[df2_gr['run_identifier'].isin(np.random.choice(df2_gr['run_identifier'].unique(), 6, replace=False))]['nb_transistors'].iloc[:3].mean()
#     )
#



df_dict = {}
# synth_out = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_3007_n_flips/synth_out_16k_done/'
synth_out = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_3007_n_flips/synth_out/'


series_list = []

for d in os.listdir(synth_out):
    try:
        df_dict[d] = pd.read_parquet(synth_out + d + '/flowy_data_record.parquet')
        df_gr = df_dict[d].groupby('step')['nb_transistors'].mean().reset_index().rename(columns={'nb_transistors': d}).set_index('step')
        series_list.append(df_gr)
    except:
        print(f'skip {d}')

df_conc = pd.concat(series_list, axis=1)


ax = df_conc.plot()
ax.set_ylim(450, 800)
# ax.get_legend().remove()
plt.show()

df_dict = {}
synth_out = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_3007_n_flips/synth_out/'

series_list = []

for d in os.listdir(synth_out):
    try:
        df_dict[d] = pd.read_parquet(synth_out + d + '/flowy_data_record.parquet')
        df_dict[d]['cummin'] = df_dict[d].groupby("run_identifier")['nb_transistors'].cummin()

        df_gr = df_dict[d].groupby('step')['cummin'].min().reset_index().rename(columns={'nb_transistors': d}).set_index('step')
        series_list.append(df_gr)
    except:
        print(f'skip {d}')

df_conc = pd.concat(series_list, axis=1)






ax = df_conc.plot()
ax.set_ylim(350, 425)
ax.get_legend().remove()
plt.show()

tc_list = ['res_00000000000011', 'res_00000000000004', 'res_00000000000017', 'res_00000000000023', 'res_00000000000016', 'res_00000000000000', 'res_00000000000006', 'res_00000000000010']
sme_list = ['res_00000000000001', 'res_00000000000005', 'res_00000000000009', 'res_00000000000015', 'res_00000000000012', 'res_00000000000020', 'res_00000000000019', 'res_00000000000007']
three_list = ['res_00000000000018', 'res_00000000000021', 'res_00000000000014', 'res_00000000000022', 'res_00000000000008', 'res_00000000000003', 'res_00000000000002', 'res_00000000000013']


# tc_list = ['res_00000000000014', 'res_00000000000007', 'res_00000000000019', 'res_00000000000017', 'res_00000000000013', 'res_00000000000000', 'res_00000000000003', 'res_00000000000011']
# sme_list = ['res_00000000000018', 'res_00000000000022', 'res_00000000000009', 'res_00000000000004', 'res_0000000000012', 'res_00000000000021', 'res_00000000000008', 'res_00000000000002']
# three_list = ['res_00000000000010', 'res_00000000000020', 'res_00000000000005', 'res_00000000000015', 'res_00000000000001', 'res_00000000000016', 'res_00000000000023', 'res_0000000000006']


ax = df_conc.iloc[:, 18:24].plot()
ax.set_ylim(450, 800)
plt.show()




data_list = []

for d in os.listdir(synth_out):
    try:
        df_dict[d] = pd.read_parquet(synth_out + d + '/flowy_data_record.parquet')

        df_6000 = df_dict[d][df_dict[d]['step']<6000].reset_index(drop=True)

        mean_min_6000 = df_6000.groupby('run_identifier')['nb_transistors'].min().mean()
        min_6000 = df_6000['nb_transistors'].min()

        mean_min = df_dict[d].groupby('run_identifier')['nb_transistors'].min().mean()
        min_ = df_dict[d]['nb_transistors'].min()


        data_list.append({
            'd': d,
            'mean_min_6000': mean_min_6000,
            'min_6000': min_6000,
            'mean_min': mean_min,
            'min': min_,
        })
    except:
        print(f'skip {d}')


data_df = pd.DataFrame(data_list)

data_df['type'] = '3007'

cond = data_df['d'].isin(tc_list)
data_df.loc[cond, 'type'] = 'tc'

cond = data_df['d'].isin(sme_list)
data_df.loc[cond, 'type'] = 'sme'


data_df = data_df.sort_values(['type']).reset_index(drop=True)



#
# df_dict['res_00000000000020']['run_identifier'].unique().shape


import bz2

path = "/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_3007_n_flips/generation_out/res_00000000000020/hdl/mydesign_comb.v.bz2"

with bz2.open(path, "rt") as f:   # "rt" = read text
    content = f.read()


import bz2

path = "/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/tc_sme_3007_n_flips/generation_out/res_00000000000009/hdl/mydesign_comb.v.bz2"

with bz2.open(path, "rt") as f:   # "rt" = read text
    content = f.read()