
import os
import pandas as pd


synth_out = (
    '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy'
    '/flowy_trans_run_12chains_3000steps_gen_iter0/synth_out/'
)


des_dirs = os.listdir(synth_out)


synth_df_dic = {
    d: pd.read_parquet(synth_out + d + f'/flowy_data_record.parquet')
    for d in des_dirs
}


best_trans_by_des = {k: v['nb_transistors'].min() for k, v in synth_df_dic.items()}
best_trans_by_des2 = {k[4:]: v['nb_transistors'].min() for k, v in synth_df_dic.items()}

analysis_df = pd.read_parquet(
    '/scratch/ramaudruz/proj/GENIAL/output/multiplier_4bi_8bo_permuti_flowy'
    '/flowy_trans_run_12chains_3000steps_gen_iter0/analysis_out/synth_analysis.db.pqt'
)


analysis_df['nb_transistors2'] = analysis_df['design_number'].map(best_trans_by_des2)

(analysis_df['nb_transistors2'] != analysis_df['nb_transistors']).sum()



analysis_df2 = analysis_df[analysis_df['nb_transistors2'] != analysis_df['nb_transistors']]





