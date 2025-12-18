
import os
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd


data_dir = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'


res_list = os.listdir(data_dir)

suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()

print(suc_list)

# df = pd.read_parquet('/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/res_00000000000000/flowy_data_record.parquet')


count_list = [pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0] for d in suc_list]


d = suc_list[0]



df = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')





data_dir = '/scratch/mbouvier/proj/output_dgfe/output/multiplier_4bi_8bo_permuti_flowy/flowy_run_12chains_3000steps_proto_iter15/synth_out/'


res_list = os.listdir(data_dir)

suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()

print(suc_list)

# df = pd.read_parquet('/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/res_00000000000000/flowy_data_record.parquet')


count_list = [pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0] for d in suc_list]


d = suc_list[0]
