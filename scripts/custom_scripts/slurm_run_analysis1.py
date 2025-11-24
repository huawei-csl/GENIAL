import pandas as pd
import numpy as np
import os



root_dir = '/scratch/ramaudruz/proj/genial/output/multiplier_4bi_8bo_permuti_allcells_notech_normal_only/'


initia_dir = 'loop_synth_gen_iter0'


df = pd.read_parquet(f'{root_dir}{initia_dir}/analysis_out/synth_analysis.db.pqt')

print(f'initial: count {len(df)} - min {df['nb_transistors'].min()} - mean {df['nb_transistors'].mean()}')

proto_dir = 'loop_synth_proto_iter'


for i in range(6):
    df = pd.read_parquet(f'{root_dir}{proto_dir}{i}/analysis_out/synth_analysis.db.pqt')
    print(f'{i}: count {len(df)} - min {df['nb_transistors'].min()} - mean {df['nb_transistors'].mean()}')


data_dir = '/scratch/ramaudruz/proj/genial/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out/'



len(os.listdir(data_dir))







