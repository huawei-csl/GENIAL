
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gen_dir_proto = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out_copy_260415b/'
syn_dir_proto = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_copy_260415b/'


diff = list(set(os.listdir(gen_dir_proto))-set(os.listdir(syn_dir_proto)))


for f in diff:
    shutil.rmtree(os.path.join(gen_dir_proto, f))


for f in os.listdir(syn_dir_proto):
    os.rename(gen_dir_proto + f, gen_dir_proto + f.replace('res_0', 'res_1'))
    os.rename(syn_dir_proto + f, syn_dir_proto + f.replace('res_0', 'res_1'))


len(os.listdir(syn_dir_proto))

gen_dir_t = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out_T/'
syn_dir_t = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_T/'


np.random.seed(42)
to_delete = np.random.choice(os.listdir(gen_dir_t), len(os.listdir(syn_dir_proto)), replace=False)

print(len(set(os.listdir(gen_dir_t)) - set(to_delete)))
print(len(set(os.listdir(syn_dir_t)) - set(to_delete)))


for f in to_delete:
    try:
        shutil.rmtree(gen_dir_t + f)
    except Exception as e:
        print(e)
    try:
        shutil.rmtree(syn_dir_t + f)
    except Exception as e:
        print(e)


gen_dir_12 = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out_12900/'
syn_dir_12 = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_12900/'

print(len(os.listdir(gen_dir_12)))
print(len(os.listdir(syn_dir_12)))

print(len(set(os.listdir(syn_dir_12)) - set(os.listdir(gen_dir_12))))
print(len(set(os.listdir(gen_dir_12)) - set(os.listdir(syn_dir_12))))




gen_dir_proto_b = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/generation_out_copy_260415b/'
syn_dir_proto_b = '/home/ramaudruz/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out_copy_260415b/'


diff = list(set(os.listdir(gen_dir_proto_b))-set(os.listdir(syn_dir_proto_b)))


for f in diff:
    shutil.rmtree(os.path.join(gen_dir_proto_b, f))


for f in os.listdir(syn_dir_proto_b):
    os.rename(gen_dir_proto_b + f, gen_dir_proto_b + f.replace('res_0', 'res_1'))
    os.rename(syn_dir_proto_b + f, syn_dir_proto_b + f.replace('res_0', 'res_1'))


data_list = []
for f in os.listdir(syn_dir_proto_b):
    try:
        dfb = pd.read_parquet(syn_dir_proto_b + f'{f}/flowy_data_record.parquet')
    except:
        shutil.rmtree(syn_dir_proto_b + f)


    if f in os.listdir(syn_dir_proto):
        try:
            dfb = pd.read_parquet(syn_dir_proto_b + f'{f}/flowy_data_record.parquet')
            df = pd.read_parquet(syn_dir_proto + f'{f}/flowy_data_record.parquet')
            data_list.append({
                'min': df.groupby('run_identifier')['nb_transistors'].min().mean(),
                'min_b': dfb.groupby('run_identifier')['nb_transistors'].min().mean(),
            })
        except:
            print(f)




df100 = pd.DataFrame(data_list)

df100.plot.scatter(x='min', y='min_b')

plt.show()



