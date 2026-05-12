
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/analysis_out/synth_analysis.db.pqt')

# get global min/max across all classes
vmin = df["nb_transistors"].min()
vmax = df["nb_transistors"].max()

# create integer-aligned bins
bins = np.arange(vmin - 0.5, vmax + 1.5, 1)

df_pivot = [df[df["design_number"].map(lambda x: x.startswith(f'{i}'))]["nb_transistors"] for i in range(4)]

plt.figure(figsize=(15, 9))
plt.hist(df_pivot, bins=bins, stacked=True, label=range(4))
plt.legend()
plt.show()



df_sorted = df.sort_values(by=["nb_transistors"]).reset_index(drop=True)


df1 = pd.read_csv('/home/ramaudruz/data_dir/misc/check_mse_260505/chunk1_pred.csv')
df1['inference_round'] = 0
df2 = pd.read_csv('/home/ramaudruz/data_dir/misc/check_mse_260505/chunk2_pred.csv')
df2['inference_round'] = 1



df_all = pd.concat([df1, df2], ignore_index=True)
df_all['pred'] = df_all['pred'].map(lambda x: eval(x)[0])

df_all['mse'] = (df_all['pred'] - df_all['target'])**2

df_all2 = df_all.sort_values(by=['mse'], ascending=False).reset_index(drop=True)

req_len = len(str(df_all2['des_num'].max()))

df_all2['actual_des_num'] = df_all2['des_num'].map(lambda x: 'res_' + (req_len - len(str(x))) * '0' + str(x))

for d in df_all2['actual_des_num'].iloc[:100].tolist():

    shutil.rmtree("/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/" + d)
    shutil.rmtree("/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/" + d)







import json

with open('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/trainer_out/dataset_split_part1.json') as f:
    d = json.load(f)


# with open('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/trainer_out/dataset_split_part2.json') as f:
#     d2 = json.load(f)

with open('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/trainer_out/dataset_split.json') as f:
    d3 = json.load(f)




#
# len(set(d['train_design_numbers']) & set(d3['train_design_numbers']))


df1 = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/analysis_out_100check/synth_analysis.db.pqt')


df2 = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/analysis_out/synth_analysis.db.pqt')

df2b = df2[df2['design_number'].isin(df1['design_number'].tolist())].reset_index(drop=True)


df2c = df2b.sort_values('design_number').reset_index(drop=True)
df1c = df1.sort_values('design_number').reset_index(drop=True)

import matplotlib.pyplot as plt

plt.scatter(df2c['nb_transistors'], df1c['nb_transistors'])

# get limits that cover both series
min_val = min(df2c['nb_transistors'].min(), df1c['nb_transistors'].min())
max_val = max(df2c['nb_transistors'].max(), df1c['nb_transistors'].max())

# plot x = y line
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

plt.xlabel("First round")
plt.ylabel("Second round")
plt.show()



import numpy as np

s1 = df1c['nb_transistors']
s2 = df2c['nb_transistors']

x, y = s1.align(s2)
mask = x.notna() & y.notna()
x, y = x[mask], y[mask]

mean_x, mean_y = x.mean(), y.mean()
var_x, var_y = x.var(), y.var()
cov_xy = np.cov(x, y, bias=True)[0, 1]

ccc = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y) ** 2)


new_data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'

within_var_list = []
mean_list = []
d_list = []
for d in os.listdir(new_data_dir):
    try:
        df = pd.read_parquet(new_data_dir + f"{d}/flowy_data_record.parquet")
        df_gr = df.groupby('run_identifier')['nb_transistors'].min()

        within_var_list.append(df_gr.var())
        mean_list.append(df_gr.mean())
        d_list.append(d)
    except:
        print(d)


sample_var = np.var(mean_list)
within_var = np.mean(within_var_list)

k = 6

sample_var / (sample_var + within_var / 6)



stats_df = pd.DataFrame({'within_var_list': within_var_list, 'mean_list': mean_list})

stats_df['coefficient_variation_of_mean'] = np.sqrt(stats_df['within_var_list']) / np.sqrt(6) / stats_df['mean_list']



stats_df['run_required'] = (np.sqrt(stats_df['within_var_list']) / stats_df['mean_list'] / 0.05) ** 2

stats_df['d'] = d_list


stats_df['run_required_decision'] = 0


cond = stats_df['run_required'] > 6
stats_df.loc[cond, 'run_required_decision'] = 1

cond = stats_df['run_required'] > 7
stats_df.loc[cond, 'run_required_decision'] = 2

cond = stats_df['run_required'] > 8
stats_df.loc[cond, 'run_required_decision'] = 3

cond = stats_df['run_required'] > 9
stats_df.loc[cond, 'run_required_decision'] = 4

cond = stats_df['run_required'] > 10
stats_df.loc[cond, 'run_required_decision'] = 5

cond = stats_df['run_required'] > 11
stats_df.loc[cond, 'run_required_decision'] = 6



stats_df.to_csv('/mnt/nvme/data_dir/misc/genial_additional_runs_required/runs_required.csv', index=False)


