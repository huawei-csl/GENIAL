
import os
import shutil
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd


data_dir = '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'


res_list = os.listdir(data_dir)

suc_list = [d for d in os.listdir(data_dir) if len(os.listdir(f'{data_dir}{d}')) > 0]
suc_list.sort()

print(suc_list)


# df = pd.read_parquet('/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/res_00000000000000/flowy_data_record.parquet')


count_list = [pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0] for d in suc_list]

count_dic = {d: pd.read_parquet(f'{data_dir}{d}/flowy_data_record.parquet')['run_identifier'].unique().shape[0] for d in suc_list}

count_dic2 = {d: c for d, c in count_dic.items() if c == 12}

count_dic3 = {d: c for d, c in count_dic.items() if c != 12}

# success_dir = list(count_dic2.keys())
# success_dir.sort()
#
# unsuccess_dir = list(count_dic3.keys())
# unsuccess_dir.sort()


print(f"Completed: {sum([c == 12 for c in count_list])}")
print(f"Incompleted: {sum([c != 12 for c in count_list])}")

d = suc_list[0]


count_dic_1 = {d: c for d, c in count_dic.items() if c == 1}
count_dic_2 = {d: c for d, c in count_dic.items() if c == 2 or c == 3}

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


