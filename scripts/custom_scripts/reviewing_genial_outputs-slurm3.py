
import os
from concurrent.futures.thread import ThreadPoolExecutor
import pandas as pd
import numpy as np


data_dir = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'



existing = pd.read_csv('/home/ramaudruz/misc/swact_data.csv')

existing2 = existing[existing['min_val'].notnull()].reset_index(drop=True)

existing2['root'] = existing['file'].map(lambda x: x.split('/synth_out/')[0])
existing2['des_number'] = existing['file'].map(lambda x: x.split('/synth_out/')[1].split('/')[0][4:])

existing2['root_des_comb'] = existing2['root'] + '_' + existing2['des_number']


existing2 = existing2.sort_values('root_des_comb').reset_index(drop=True)

split_df = {}

for r in existing2['root'].unique():
    split_df[r] = existing2[existing2['root'] == r].reset_index(drop=True)

missing_dirs = []
for r, df in split_df.items():
    if not os.path.exists(f'{r}/analysis_out/gener_analysis.db.pqt'):
        missing_dirs.append(r)


    info_df = pd.read_parquet(f'{r}/analysis_out/gener_analysis.db.pqt')
    info_df2 = info_df[info_df['design_number'].isin(df['des_number'].unique().tolist())].reset_index(drop=True)
    temp_map = dict(zip(info_df2['design_number'], info_df2['encodings_input']))
    df['encodings_input'] = df['des_number'].map(temp_map)




pd.concat(split_df.values(), ignore_index=True).to_csv('/home/ramaudruz/misc/swact_data_with_encoding.csv', index=False)


existing3 = existing2[~existing2['root'].isin(missing_dirs)].reset_index(drop=True)

existing3.to_csv('/home/ramaudruz/misc/swact_data_with_ref.csv', index=False)


# split_df = {}
#
# for r in existing3['root'].unique():
#     split_df[r] = existing3[existing3['root'] == r].reset_index(drop=True)
#
# missing_dirs = []
# for r, df in split_df.items():
#     if not os.path.exists(f'{r}/analysis_out/gener_analysis.db.pqt'):
#         missing_dirs.append(r)
#
#
#     info_df = pd.read_parquet(f'{r}/analysis_out/gener_analysis.db.pqt')
#     info_df2 = info_df[info_df['design_number'].isin(df['des_number'].unique().tolist())].reset_index(drop=True)
#     temp_map = dict(zip(info_df2['design_number'], info_df2['encodings_input']))
#     df['encodings_input'] = df['des_number'].map(temp_map)
#
# pd.concat(split_df.values(), ignore_index=True).to_csv('/home/ramaudruz/misc/swact_data_with_encoding.csv', index=False)
#


exiting_10 = pd.read_csv('/home/ramaudruz/misc/swact_data_with_encoding.csv')



exiting_10['hist_value'] = pd.cut(
    exiting_10['min_val'],
    bins=100
)

vc = exiting_10['hist_value'].value_counts().reset_index()

vc['500'] = 500
vc['min_count'] = np.minimum(vc['500'], vc['count'])


vc2 = vc.sort_values('hist_value').reset_index(drop=True)

sampled_df = (
    exiting_10
    .groupby('hist_value', group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), 300), replace=False))
)

sampled_df.to_csv('/home/ramaudruz/misc/swact_data_with_encoding-selected.csv', index=False)

#
# enc_map = {}
# for i, row in existing2.iterrows():
#     row['file']
#     break
#
#
# existing2
#
#
#
# df10 = pd.read_parquet(f'/scratch/mbouvier/proj/output_dgfe/output/multiplier_4bi_8bo_permuti_flowy/standard_flowy_with_sinkhorn_proto_iter11/analysis_out/gener_analysis.db.pqt')
#



