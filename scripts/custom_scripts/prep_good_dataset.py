import os
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_cache/'

data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'
gen_data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'


data_dir2 = '/scratch/ramaudruz/synth_out_260211/'

suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()


count_dic = {}
count_dic2 = {}

for i, d in enumerate(suc_list):
    if i % 100 == 0:
        print(i)
    df = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')

    count_dic[d] = df['run_identifier'].unique().shape[0]

    cond = df['runtime_full_mockturtle_step'] > 1000
    count_dic2[d] = df[cond]['run_identifier'].unique().shape[0]



print(f"Completed: {sum([c == 6 for c in count_dic.values()])}")
print(f"Incompleted: {sum([c != 6 for c in count_dic.values()])}")


pd.Series(count_dic.values()).value_counts()
pd.Series(count_dic2.values()).value_counts()


set(os.listdir(data_dir)) & set(os.listdir(data_dir2))



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


# data_list = []
#
# mean_df_dic = {}
#
# count_dic2 = {k: v for k, v in count_dic.items() if v >=5}
# for d in count_dic2:
#     df = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')
#     mean_df_dic[d] = df.groupby('step')['nb_transistors'].min().reset_index()
#
# df_conc = pd.concat([v.set_index('step')[['nb_transistors']].rename(columns={'nb_transistors': d}) for d, v in mean_df_dic.items()], axis=1)
#
# ax = df_conc.plot()
# ax.set_ylim(400, 1800)
# ax.set_xlim(30000, 31000)
# plt.show()



#
# to_del = []
#
# synth_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'
#
#
# for d in os.listdir(synth_dir):
#     if d not in to_keep:
#         to_del.append(d)
#         shutil.rmtree(f'{synth_dir}/{d}')
#
#
#
#
#
# anal_df = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/analysis_out/synth_analysis.db.pqt').sort_values('design_number').reset_index(drop=True)
#

data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0_old2/synth_out_CACHE/'


suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()

count_dic = {}

data_list = []

for i, d in enumerate(suc_list):
    if i % 100 == 0:
        print(i)
    # count_dic[d] = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0]

    mean_min = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet').groupby('run_identifier')['nb_transistors'].min().mean()

    # cond1 = flowy_df['run_identifier'].isin(flowy_df['run_identifier'].unique().tolist()[:6])
    # cond2 = flowy_df['step'] >= 2000
    #
    # flowy_df2 = flowy_df[cond1 & cond2].reset_index(drop=True)
    # std = flowy_df2.groupby('run_identifier')["nb_transistors"].min().std()

    data_list.append({'d': d, 'mean_min': mean_min})


anal_df10 = pd.DataFrame(data_list)

anal_df10['mean_min_high_effort'] = anal_df10['d'].map(mean_min_dic)


data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'


suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()

count_dic = {}

data_list = []

for i, d in enumerate(suc_list):
    if i % 100 == 0:
        print(i)
    # count_dic[d] = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0]

    mean_min = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet').groupby('run_identifier')['nb_transistors'].min().mean()

    # cond1 = flowy_df['run_identifier'].isin(flowy_df['run_identifier'].unique().tolist()[:6])
    # cond2 = flowy_df['step'] >= 2000
    #
    # flowy_df2 = flowy_df[cond1 & cond2].reset_index(drop=True)
    # std = flowy_df2.groupby('run_identifier')["nb_transistors"].min().std()

    data_list.append({'d': d, 'mean_min': mean_min})


anal_df20 = pd.DataFrame(data_list)

strong_effort_dic = dict(zip(anal_df20['d'], anal_df20['mean_min']))


anal_df10['mean_min_high_effort'] = anal_df10['d'].map(strong_effort_dic)

anal_df11 = anal_df10[anal_df10['mean_min_high_effort'].notnull()].reset_index(drop=True).rename(columns={'mean_min': 'Mean min MEDIUM EFFORT', 'mean_min_high_effort': 'Mean min HIGH EFFORT'})

x_line = np.linspace(anal_df11['Mean min MEDIUM EFFORT'].min(), anal_df11['Mean min MEDIUM EFFORT'].max(), 500)
y_line = x_line

fig, axes = plt.subplots(1, 1, figsize=(12, 8))
anal_df11.plot.scatter(
    x='Mean min MEDIUM EFFORT',
    y='Mean min HIGH EFFORT',
    ax=axes
)
axes.plot(x_line, y_line, color='black')

axes.set_xlabel('Mean min MEDIUM EFFORT')
axes.set_ylabel('Mean min HIGH EFFORT')

plt.tight_layout()   # 🔑 prevents label cut-off
plt.show()

# #
# fig, axes = plt.subplots(1, 1, figsize=(16, 16))
# anal_df11.plot.scatter(x='Mean min MEDIUM EFFORT', y='Mean min HIGH EFFORT')
# plt.plot(x_line, y_line, color='black')
# plt.xlabel('Mean min MEDIUM EFFORT')
# plt.xlabel('Mean min HIGH EFFORT')
# plt.show()
fig, axes = plt.subplots(1, 1, figsize=(12, 8))

anal_df11.plot.scatter(
    x='Mean min MEDIUM EFFORT',
    y='Mean min HIGH EFFORT',
    ax=axes
)

axes.plot(x_line, y_line, color='black')

axes.set_xlabel('Mean min MEDIUM EFFORT')
axes.set_ylabel('Mean min HIGH EFFORT')

plt.tight_layout()   # 🔑 prevents label cut-off
plt.show()


anal_df12 = anal_df11[anal_df11['Mean min MEDIUM EFFORT']>=700].reset_index(drop=True)

x_line = np.linspace(anal_df12['Mean min MEDIUM EFFORT'].min(), anal_df12['Mean min MEDIUM EFFORT'].max(), 500)
y_line = x_line

anal_df12.plot.scatter(x='Mean min MEDIUM EFFORT', y='Mean min HIGH EFFORT')
plt.plot(x_line, y_line)
plt.show()

cond = (
    (anal_df11['Mean min MEDIUM EFFORT'] > 1325) &
    (anal_df11['Mean min MEDIUM EFFORT'] < 1375)
)
anal_df11_1350 = anal_df11[cond].reset_index(drop=True)

to_keep = ['res_00000000000118', 'res_00000000000418', 'res_00000000000016', 'res_00000000000156', 'res_00000000000181', 'res_00000000001444', 'res_00000000001036', 'res_00000000002684']

gen_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'

for d in os.listdir(gen_dir):
    if d not in to_keep:
        shutil.rmtree(gen_dir + d)

cond = (
        (anal_df11['Mean min MEDIUM EFFORT'] > 950) &
        (anal_df11['Mean min MEDIUM EFFORT'] < 1050)
)
anal_df11_1000 = anal_df11[cond].reset_index(drop=True)

cond = (
        (anal_df11['Mean min MEDIUM EFFORT'] > 750) &
        (anal_df11['Mean min MEDIUM EFFORT'] < 850)
)
anal_df11_800 = anal_df11[cond].reset_index(drop=True)

cond = (
        (anal_df11['Mean min MEDIUM EFFORT'] > 100) &
        (anal_df11['Mean min MEDIUM EFFORT'] < 500)
)
anal_df11_500 = anal_df11[cond].reset_index(drop=True)

anal_df12['requires_high_effort'] = anal_df12['Mean min MEDIUM EFFORT'] > anal_df12['Mean min HIGH EFFORT']

import matplotlib.pyplot as plt

mask = anal_df12["requires_high_effort"]  # boolean column

plt.scatter(anal_df12.loc[~mask, "Mean min MEDIUM EFFORT"], anal_df12.loc[~mask, "Mean min HIGH EFFORT"], label="low_effort")
plt.scatter(anal_df12.loc[mask,  "Mean min MEDIUM EFFORT"], anal_df12.loc[mask,  "Mean min HIGH EFFORT"], label="requires_high_effort")

plt.legend()
plt.show()

anal_df100 = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0_old2/analysis_out/synth_analysis_long.db.pqt').sort_values('design_number').reset_index(drop=True)


anal_df100['d'] = 'res_' + anal_df100['design_number']

import torch
anal_df100['input_matrix'] = anal_df100['encodings_input'].map(lambda x: torch.from_numpy(np.concatenate([np.array([[int(i) for i in v]]) for v in eval(x).values()], axis=0)))


matrix_map = dict(zip(anal_df100['d'], anal_df100['input_matrix']))
encoding_input_map = dict(zip(anal_df100['d'], anal_df100['encodings_input']))

anal_df12['input_matrix'] = anal_df12['d'].map(matrix_map)
anal_df12['encodings_input'] = anal_df12['d'].map(encoding_input_map)

all_encodings = torch.stack(anal_df12['input_matrix'].tolist())

def adjacent_hamming_cost(X: torch.Tensor) -> torch.Tensor:
    return (X[:-1] - X[1:]).abs().sum()

anal_df12['hamming_cost'] = anal_df12['input_matrix'].map(lambda x: adjacent_hamming_cost(x))


sc = plt.scatter(
    anal_df12["Mean min MEDIUM EFFORT"],
    anal_df12["Mean min HIGH EFFORT"],
    c=anal_df12["hamming_cost"],      # e.g. 0,1,2,3 or 1–5
    cmap="viridis",
    alpha=0.6
)

plt.colorbar(sc, label="hamming_cost")
plt.show()


matrix = anal_df12['input_matrix'].iloc[0]


def total_adj_diff(matrix: torch.Tensor) -> float:
    """
    Sum of Hamming distances between consecutive rows.
    """
    diffs = (matrix[1:] != matrix[:-1]).sum()
    return diffs.item()  # scalar

anal_df12['total_adj_diff'] = anal_df12['input_matrix'].map(lambda x: total_adj_diff(x))


sc = plt.scatter(
    anal_df12["Mean min MEDIUM EFFORT"],
    anal_df12["Mean min HIGH EFFORT"],
    c=anal_df12["total_adj_diff"],      # e.g. 0,1,2,3 or 1–5
    cmap="viridis",
    alpha=0.6
)

plt.colorbar(sc, label="total_adj_diff")
plt.show()

def total_column_flips(matrix: torch.Tensor) -> float:
    """
    Counts the number of bit flips per column between consecutive rows, summed over all columns.
    """
    flips = (matrix[1:] != matrix[:-1]).sum(dim=0)
    return flips.sum().item()

anal_df12['total_column_flips'] = anal_df12['input_matrix'].map(lambda x: total_column_flips(x))


sc = plt.scatter(
    anal_df12["Mean min MEDIUM EFFORT"],
    anal_df12["Mean min HIGH EFFORT"],
    c=anal_df12["total_column_flips"],      # e.g. 0,1,2,3 or 1–5
    cmap="viridis",
    alpha=0.6
)

plt.colorbar(sc, label="total_column_flips")
plt.show()

from scipy.stats import kendalltau, spearmanr
def kendall_inversions(matrix: torch.Tensor) -> float:
    """
    Maps each row to its index in lexicographically sorted order
    and computes Kendall tau distance to perfect order.
    Returns 1 - tau (closer to 1 = more inversions).
    """
    rows = [tuple(row.tolist()) for row in matrix]
    canonical = sorted(rows)
    perm_indices = [canonical.index(row) for row in rows]
    tau, _ = kendalltau(list(range(len(perm_indices))), perm_indices)
    return float(1 - tau)


anal_df12['kendall_inversions'] = anal_df12['input_matrix'].map(lambda x: kendall_inversions(x))


sc = plt.scatter(
    anal_df12["Mean min MEDIUM EFFORT"],
    anal_df12["Mean min HIGH EFFORT"],
    c=anal_df12["kendall_inversions"],      # e.g. 0,1,2,3 or 1–5
    cmap="viridis",
    alpha=0.6
)

plt.colorbar(sc, label="kendall_inversions")
plt.show()

def spearman_correlation(matrix: torch.Tensor) -> float:
    """
    Spearman rank correlation between current row order and canonical order.
    """
    rows = [tuple(row.tolist()) for row in matrix]
    canonical = sorted(rows)
    perm_indices = [canonical.index(row) for row in rows]
    corr, _ = spearmanr(list(range(len(perm_indices))), perm_indices)
    return float(corr)

anal_df12['spearman_correlation'] = anal_df12['input_matrix'].map(lambda x: spearman_correlation(x))


sc = plt.scatter(
    anal_df12["Mean min MEDIUM EFFORT"],
    anal_df12["Mean min HIGH EFFORT"],
    c=anal_df12["spearman_correlation"],      # e.g. 0,1,2,3 or 1–5
    cmap="viridis",
    alpha=0.6
)

plt.colorbar(sc, label="spearman_correlation")
plt.show()

def heavy_row_index_sum(matrix: torch.Tensor) -> float:
    """
    Sum over rows of (row index * row sum)
    Captures whether 'heavier' rows appear early or late.
    """
    row_sums = matrix.sum(dim=1)
    indices = torch.arange(matrix.size(0))
    return float((indices * row_sums).sum())


anal_df12['heavy_row_index_sum'] = anal_df12['input_matrix'].map(lambda x: heavy_row_index_sum(x))


sc = plt.scatter(
    anal_df12["Mean min MEDIUM EFFORT"],
    anal_df12["Mean min HIGH EFFORT"],
    c=anal_df12["heavy_row_index_sum"],      # e.g. 0,1,2,3 or 1–5
    cmap="viridis",
    alpha=0.6
)

plt.colorbar(sc, label="heavy_row_index_sum")
plt.show()

from scipy.stats import entropy
def row_position_entropy(matrix: torch.Tensor) -> float:
    """
    Computes entropy based on row sums.
    """
    row_sums = matrix.sum(dim=1).float()
    probs = row_sums / row_sums.sum()
    probs = probs[probs > 0]  # remove zero entries to avoid log(0)
    return float(entropy(probs.cpu().numpy(), base=2))


anal_df12['row_position_entropy'] = anal_df12['input_matrix'].map(lambda x: row_position_entropy(x))


sc = plt.scatter(
    anal_df12["Mean min MEDIUM EFFORT"],
    anal_df12["Mean min HIGH EFFORT"],
    c=anal_df12["row_position_entropy"],      # e.g. 0,1,2,3 or 1–5
    cmap="viridis",
    alpha=0.6
)

plt.colorbar(sc, label="row_position_entropy")
plt.show()

def heavy_row_position(matrix: torch.Tensor) -> float:
    row_weights = matrix.sum(dim=1)
    indices = torch.arange(matrix.size(0))
    return float((row_weights * indices).sum() / row_weights.sum())


anal_df12['heavy_row_position'] = anal_df12['input_matrix'].map(lambda x: heavy_row_position(x))


sc = plt.scatter(
    anal_df12["Mean min MEDIUM EFFORT"],
    anal_df12["Mean min HIGH EFFORT"],
    c=anal_df12["heavy_row_position"],      # e.g. 0,1,2,3 or 1–5
    cmap="viridis",
    alpha=0.6
)

plt.colorbar(sc, label="heavy_row_position")
plt.show()


def pairwise_order_signature(matrix: torch.Tensor) -> torch.Tensor:
    """
    Returns a 16x16 matrix P where P[i,j] = 1 if row i appears before row j.
    Flatten this to a 256-dim vector to use as a feature.
    """
    rows = [tuple(row.tolist()) for row in matrix]
    canonical = sorted(rows)  # assign row IDs 0-15
    row_ids = torch.tensor([canonical.index(r) for r in rows])  # row_id -> position
    n = len(row_ids)
    P = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if row_ids[i] < row_ids[j]:
                P[i,j] = 1
    return P.flatten()

anal_df12['pairwise_order_signature'] = anal_df12['input_matrix'].map(lambda x: pairwise_order_signature(x))


anal_df13 = anal_df12.sort_values("Mean min MEDIUM EFFORT").reset_index(drop=True)

anal_df13a = anal_df13[anal_df13['requires_high_effort']].iloc[:6]
anal_df13b = anal_df13[~anal_df13['requires_high_effort']].iloc[:6]

import genial.experiment.plotter as plotter

fig, axes = plt.subplots(1, 1, figsize=(10, 10))
plotter.plot_encoding_heatmap_solo(
    ax=axes,
    encoding_str=str(encodings_input_list[j]),
    design_number="__(^^)__",
    bitwidth=4,
    port_type="input",
    ax_title=f"",
)
plt.show()


import umap


#
# pairwise_features = torch.stack([pairwise_order_signature(m) for m in anal_df12['pairwise_order_signature']])
# Convert to numpy for UMAP
pairwise_features_np = torch.cat([x.reshape(1, -1) for x in anal_df12['pairwise_order_signature']], axis=0).numpy()

# --- Step 3: Run UMAP ---
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(pairwise_features_np)

# --- Step 4: Scatter plot ---
# If you have cluster labels for coloring, use 'labels' array
# Otherwise just plot
plt.figure(figsize=(6,6))
plt.scatter(embedding[:,0], embedding[:,1], c=anal_df12["requires_high_effort"], cmap='tab10', s=50)
plt.title("UMAP of Pairwise Order Signatures")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()




anal_df12.shape


X = anal_df12['input_matrix'].tolist()
y = anal_df12['requires_high_effort'].tolist()
encodings_input_list = anal_df12['encodings_input'].tolist()

def hamming(a, b):
    return (a != b).sum().item()

for i in range(len(X)):
    xi, yi = X[i], y[i]
    candidates = [(j, hamming(xi, X[j]))
                  for j in range(len(X)) if y[j] != yi]
    j_min, d_min = min(candidates, key=lambda t: t[1])
    print(i, "closest opposite-cluster distance:", d_min, anal_df12['Mean min MEDIUM EFFORT'].iloc[i])
    # if d_min == 4:
    #     break



found = False

for i in range(len(X)):
    xi = X[i]
    yi = y[i]

    for j in range(len(X)):
        if y[j] == yi:
            continue

        d = hamming(xi, X[j])

        if d == 8:
            print(f"Found boundary pair!")
            print(f"Sample i: {i}, label={yi}")
            print(f"Sample j: {j}, label={y[j]}")
            print(f"Hamming distance: {d}")

            x1 = X[i]   # shape (16,4)
            x2 = X[j]   # shape (16,4)

            found = True
            break

    if i == 739:
        break


import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


anal_df12['np_arr'] = anal_df12['encodings_input'].map(lambda x: np.concatenate([np.array([int(i) for i in v]) for v in eval(x).values()]))

anal_df12_train = anal_df12.iloc[:1500]

anal_df12_test = anal_df12.iloc[1500:].reset_index(drop=True)

X_train = np.concatenate([a.reshape(1, -1) for a in anal_df12_train['np_arr']], axis=0)
y_train = np.array([int(i) for i in anal_df12_train['requires_high_effort']])
X_test = np.concatenate([a.reshape(1, -1) for a in anal_df12_test['np_arr']], axis=0)
y_test = np.array([int(i) for i in anal_df12_test['requires_high_effort']])

# --- Step 2: Train decision tree ---
clf = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)  # shallow tree for interpretability
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



print((y_pred == y_test).sum() / len(y_pred))


# --- Step 3: Inspect feature importance ---
importances = clf.feature_importances_
# Map back to (row, col)
for idx, imp in enumerate(importances):
    if imp > 0:
        row, col = divmod(idx, 4)
        print(f"Row {row}, Column {col} importance: {imp:.3f}")

# --- Step 4: Optional: visualize the tree ---
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=[f"r{r}c{c}" for r in range(16) for c in range(4)],
          class_names=[str(l) for l in np.unique(y)],
          filled=True, rounded=True, fontsize=10)
plt.show()




anal_df = pd.read_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0_old2/analysis_out/synth_analysis.db.pqt').sort_values('design_number').reset_index(drop=True)


anal_df2 = anal_df.sample(frac=1).reset_index(drop=True).iloc[:2700]
anal_df2.to_parquet('/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0_old2/analysis_out/synth_analysis_short.db.pqt', index=False)



data_dir = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0_old2/synth_out_CACHE/'






