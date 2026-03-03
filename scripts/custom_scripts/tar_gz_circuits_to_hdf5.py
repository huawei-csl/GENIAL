
import os
import h5py
import tarfile
import io
import pandas as pd
import numpy as np
import polars as pl
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def read_and_tag(round_num, mig_dir):
    parquet_file = mig_dir + f"mig_output_round_{round_num}.parquet"
    if not os.path.isfile(parquet_file):
        return None  # or raise an exception if you want to stop
    df = pd.read_parquet(parquet_file)
    df.insert(0, "round", round_num)
    return df


def check_if_done_already(run_dir_full):
    return len([f for f in os.listdir(run_dir_full) if f.startswith("mig_circuits")]) == 10


# Synth out dir
enc_dir = "/mnt/nvme/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/gnn_data_collection/synth_out/"

# Loop through encodings
for dir_count, d in enumerate(os.listdir(enc_dir)):
    # Print log
    if dir_count % 10 == 0:
        print(dir_count)
    # Retrieve run dirs
    run_dirs = [rd for rd in os.listdir(enc_dir + d) if rd.startswith("run")]
    # Loop through run dirs
    for run_count, run_dir in enumerate(run_dirs):
        # Extract mig cache tar gz
        run_dir_full = f"{enc_dir}{d}/{run_dir}/"

        # Check if completed. Then skip.
        if check_if_done_already(run_dir_full):
            continue

        # Mig dir
        mig_dir = run_dir_full + "mig_cache/"

        # Delete extracted mig dir if it already exists
        if os.path.isdir(mig_dir):
            shutil.rmtree(mig_dir)

        # Extract tar.gz
        with tarfile.open(run_dir_full + "mig_cache.tar.gz", "r:gz") as tar:
            tar.extractall(path=run_dir_full, filter=None)

        # Load all parquets
        t = time.time()

        df_list = []
        with ProcessPoolExecutor(max_workers=52) as executor:  # adjust workers as needed
            futures = {executor.submit(read_and_tag, r, mig_dir): r for r in range(10_000)}
            for future in as_completed(futures):
                df = future.result()
                if df is not None:
                    df_list.append(df)
        print(time.time() - t)

        # Skip if incomplete
        if len(df_list) != 10_000:
            continue

        # Concatenate and create combined parquet
        df_all = pd.concat(df_list, ignore_index=True)
        df_all['round'] = df_all['round'].astype('int16')
        df_all_pl = pl.from_pandas(df_all)

        for round_range in range(1_000, 10_001, 1_000):
            cond = (pl.col('round') >= round_range - 1_000) & (pl.col('round') < round_range)
            df_all_pl_temp = df_all_pl.filter(cond)
            pl_out_path = f"{enc_dir}{d}/{run_dir}/mig_circuits_{round_range}.parquet"
            df_all_pl_temp.write_parquet(pl_out_path)

        # Delete extracted mig dir
        if os.path.isdir(mig_dir):
            shutil.rmtree(mig_dir)
