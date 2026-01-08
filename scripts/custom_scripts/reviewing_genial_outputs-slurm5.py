
import os
import shutil
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd


data_dir = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_merge_iter5/synth_out/'


suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()


count_dic = {}

for i, d in enumerate(suc_list):
    if i % 100 == 0:
        print(i)
    count_dic[d] = pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0]



print(f"Completed: {sum([c == 12 for c in count_dic.values()])}")
print(f"Incompleted: {sum([c != 12 for c in count_dic.values()])}")


pd.Series(count_dic.values()).value_counts()


df = pd.read_parquet('/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_merge_iter5/analysis_out/synth_analysis.db.pqt')



