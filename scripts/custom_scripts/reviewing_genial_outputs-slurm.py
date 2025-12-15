
import os
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd

#
# synth_out = (
#     '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy'
#     '/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'
# )
#
#
# des_dirs = os.listdir(synth_out)
#
#
# synth_df_dic = {
#     d: pd.read_parquet(synth_out + d + f'/flowy_data_record.parquet')
#     for d in des_dirs
# }
#
#
# best_trans_by_des = {k: v['nb_transistors'].min() for k, v in synth_df_dic.items()}
# best_trans_by_des2 = {k[4:]: v['nb_transistors'].min() for k, v in synth_df_dic.items()}
#
# analysis_df = pd.read_parquet(
#     '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy'
#     '/flowy_trans_run_12chains_3000steps_gen_iter0/analysis_out/synth_analysis.db.pqt'
# )
#
#
# analysis_df['nb_transistors2'] = analysis_df['design_number'].map(best_trans_by_des2)
#
# (analysis_df['nb_transistors2'] != analysis_df['nb_transistors']).sum()
#
#
#
# analysis_df2 = analysis_df[analysis_df['nb_transistors2'] != analysis_df['nb_transistors']]
#
#
#
#
#
# import pandas as pd
#
#
# import pandas as pd
#
# with open('/home/ramaudruz/squeue_output.txt', 'r') as f:
#     lines = f.readlines()
#
# lines_clean = [
#     [ls for ls in l.split() if ls != '']
#     for l in lines
# ]
#
# df = pd.DataFrame(lines_clean[1:], columns=lines_clean[0] + [f'col_{i}' for i in range(3)])
#



#
# from pathlib import Path
#
# def find_files(start_dir, filename):
#     start = Path(start_dir).resolve()
#     return [str(p) for p in start.rglob(filename) if p.is_file()]
#
#
# all_files = find_files('/scratch/mbouvier/proj/output_dgfe/output/multiplier_4bi_8bo_permuti_flowy', 'flowy_data_record.parquet')
#

if __name__ == "__main__":



    root_dir = '/scratch/mbouvier/proj/output_dgfe/output/multiplier_4bi_8bo_permuti_flowy/'

    run_dirs = os.listdir(root_dir)

    flowy_files = []


    for run_dir in run_dirs:
        if run_dir.endswith('.tar.gz'):
            continue
        synth_dir = os.path.join(root_dir, f'{run_dir}/synth_out/')
        if not os.path.isdir(synth_dir):
            continue
        des_dir = os.listdir(synth_dir)
        for d in des_dir:
            parquet_file = os.path.join(synth_dir, f'{d}/flowy_data_record.parquet')
            if os.path.isfile(parquet_file):
                print('t')
                flowy_files.append(parquet_file)
            else:
                print('f')


    len(flowy_files)


    column_tuple_set = set()
    df = pd.read_parquet(flowy_files[0])

    f_to_min_swact_list = []

    for i, f in enumerate(flowy_files):
        if i % 1000 == 0:
            print(i)
        df = pd.read_parquet(f)
        min_val = df['swact_count_weighted'].min()

        f_to_min_swact_list.append({
            'file': f,
            'min_val': min_val,
        })

    pd.DataFrame(f_to_min_swact_list).to_csv('/home/ramaudruz/misc/swact_data.csv', index=False)

    #
    #
    # import pandas as pd
    # from concurrent.futures import ProcessPoolExecutor, as_completed
    # import os
    #
    # def min_swact_for_file(f):
    #     df = pd.read_parquet(f)
    #     return f, df['swact_count_weighted'].min()
    #
    # def compute_min_swact(flowy_files, max_workers=None):
    #     f_to_min_swact_list = []
    #
    #     with ThreadPoolExecutor(max_workers=16) as executor:
    #         futures = {
    #             executor.submit(min_swact_for_file, f): i
    #             for i, f in enumerate(flowy_files)
    #         }
    #
    #         for j, future in enumerate(as_completed(futures)):
    #             f, min_val = future.result()
    #             f_to_min_swact_list.append({
    #                 'file': f,
    #                 'min_val': min_val,
    #             })
    #
    #             if j % 1000 == 0:
    #                 print(j)
    #
    #     return f_to_min_swact_list
    #
    #
    # f_to_min_swact_list = compute_min_swact(flowy_files, max_workers=os.cpu_count())
    #
    # pd.DataFrame(f_to_min_swact_list).to_csv('/home/ramaudruz/misc/swact_data.csv', index=False)
    #
    #
