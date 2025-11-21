import pandas as pd
import numpy as np



root_dir = '/scratch/ramaudruz/proj/genial/output/multiplier_4bi_8bo_permuti_allcells_notech_normal_only/'


initia_dir = 'loop_synth_gen_iter0'


df = pd.read_parquet(f'{root_dir}{initia_dir}/analysis_out/synth_analysis.db.pqt')

df['nb_transistors'].mean()
df['nb_transistors'].min()

proto_dir = 'loop_synth_proto_iter'


for i in range(6):
    df = pd.read_parquet(f'{root_dir}{proto_dir}{i}/analysis_out/synth_analysis.db.pqt')
    print(f'{i}: count {len(df)} - min {df['nb_transistors'].min()} - mean {df['nb_transistors'].mean()}')



