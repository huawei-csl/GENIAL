import os
import polars as pl
import numpy as np
import time

enc_dir = "/mnt/nvme/data_dir/4bi_8bo_rnd_in_fix_out/output/multiplier_4bi_8bo_permuti_flowy/gnn_data_collection/synth_out/"

df_list = []

# -------------------------------------------------
# Collect all data
# -------------------------------------------------

for dir_count, d in enumerate(os.listdir(enc_dir)):
    if dir_count % 10 == 0:
        print(dir_count)

    run_dirs = [rd for rd in os.listdir(enc_dir + d) if rd.startswith("run")]

    for run_dir in run_dirs:
        run_dir_full = f"{enc_dir}{d}/{run_dir}/"
        flowy_parquet = run_dir_full + "flowy_record.parquet"

        if os.path.isfile(flowy_parquet):

            df = pl.read_parquet(flowy_parquet)

            df = (
                df
                .with_columns([
                    pl.lit(d).alias("enc_id"),
                    pl.col("gates").cast(pl.Int16),
                    pl.col("depth").cast(pl.Int16),
                ])
                .drop(["recipe_round", "score"])
            )

            df_list.append(df)

# Concatenate
df_all = pl.concat(df_list, how="vertical")

# -------------------------------------------------
# Feature engineering
# -------------------------------------------------

# Compute min/max
min_val = df_all.select(pl.col("gates").min()).item()
max_val = df_all.select(pl.col("gates").max()).item()

# Create 100 equal-width bins
bin_edges = np.linspace(min_val, max_val, 101)

df_all = df_all.with_columns(
    pl.col("gates")
    .cut(breaks=bin_edges[1:-1].tolist())  # internal breaks only
    .cast(pl.Utf8)
    .alias("gates_bin")
)

df_all = df_all.with_columns([
    (pl.col("gates_bin") + "_" +
     pl.col("depth").cast(pl.Utf8) + "_" +
     pl.col("enc_id").cast(pl.Utf8)
     ).alias("group_id"),

    (pl.col("gates_bin") + "_" +
     pl.col("depth").cast(pl.Utf8)
     ).alias("group_id2")
])

# -------------------------------------------------
# Value counts
# -------------------------------------------------

gate_depth_vc = (
    df_all
    .group_by("group_id2")
    .len()
)

# Target group count
target_count = 300

# Split <500 and >=500
small_df = gate_depth_vc.filter(pl.col("len") < target_count).select("group_id2")
large_df = gate_depth_vc.filter(pl.col("len") >= target_count).select("group_id2")

df_all_sub1 = df_all.join(small_df, on="group_id2", how="semi")
df_all_sub2 = df_all.join(large_df, on="group_id2", how="semi")

# -------------------------------------------------
# Encoding counts
# -------------------------------------------------

df_all_sub1_vc = (
    df_all_sub1
    .group_by("enc_id")
    .len()
)

enc_count = dict(zip(
    df_all_sub1_vc["enc_id"].to_list(),
    df_all_sub1_vc["len"].to_list()
))

# Fill missing encodings
all_encodings = df_all["enc_id"].unique().to_list()
for e in all_encodings:
    enc_count.setdefault(e, 0)

min_e_count = min(enc_count.values())

new_sample_df = None

for curr_count in range(min_e_count, target_count):

    print(f'\ncurr_count: {curr_count}')

    start_time = time.time()

    to_add = target_count - curr_count

    curr_e_map = {k: v for k, v in enc_count.items() if v == curr_count}

    print(f'{len(curr_e_map)} encodings to add.')

    # -------------------------------------------------
    # Second subset sampling
    # -------------------------------------------------



    for e in curr_e_map:

        df_all_sub2_e = df_all_sub2.filter(pl.col("enc_id") == e)

        # Shuffle
        df_all_sub2_e = df_all_sub2_e.sample(
            n=df_all_sub2_e.height,
            shuffle=True
        )

        # Drop duplicates
        df_all_sub2_e_u = df_all_sub2_e.unique(subset=["group_id"])

        if df_all_sub2_e_u.height <= to_add:
            if new_sample_df is None:
                new_sample_df = df_all_sub2_e_u
            else:
                new_sample_df = pl.concat([new_sample_df, df_all_sub2_e_u])

        else:
            if new_sample_df is None:
                new_sample_df = df_all_sub2_e_u.head(to_add)

            else:
                temp_vc = (
                    new_sample_df
                    .group_by("group_id2")
                    .len()
                )

                value_count_map = dict(zip(
                    temp_vc["group_id2"].to_list(),
                    temp_vc["len"].to_list()
                ))

                df_all_sub2_e_u = df_all_sub2_e_u.with_columns(
                    pl.col("group_id2")
                    .map_elements(lambda x: value_count_map.get(x, 0))
                    .alias("temp_count")
                )

                df_all_sub2_e_u = (
                    df_all_sub2_e_u
                    .sort("temp_count")
                    .drop("temp_count")
                )

                new_sample_df = pl.concat([
                    new_sample_df,
                    df_all_sub2_e_u.head(to_add)
                ])
    print(f'Took {int(time.time() - start_time)} seconds!')


selected_samples_df = pl.concat([
    df_all_sub1,
    new_sample_df
])


selected_samples_vc = (
    selected_samples_df
    .group_by("enc_id")
    .len()
)

last_enc_count = dict(zip(
    selected_samples_vc["enc_id"].to_list(),
    selected_samples_vc["len"].to_list()
))

# Fill missing encodings
all_encodings = df_all["enc_id"].unique().to_list()
for e in all_encodings:
    last_enc_count.setdefault(e, 0)

selected_samples_group_vc = (
    selected_samples_df
    .group_by("group_id2")
    .len()
)
last_enc_group_count = dict(zip(
    selected_samples_group_vc["group_id2"].to_list(),
    selected_samples_group_vc["len"].to_list()
))

# Fill missing encodings
all_encodings = df_all["group_id2"].unique().to_list()
for e in all_encodings:
    last_enc_group_count.setdefault(e, 0)

#############################

import matplotlib.pyplot as plt
import numpy as np

# Extract numpy arrays
gates = selected_samples_df.select("gates").to_numpy().flatten()
depth = selected_samples_df.select("depth").to_numpy().flatten()

plt.figure(figsize=(8, 6))

plt.hist2d(
    gates,
    depth,
    bins=100,
)

plt.colorbar(label="Count")
plt.xlabel("Gates")
plt.ylabel("Depth")
plt.title("2D Density of Gates vs Depth")
plt.show()


selected_samples_df_pd = selected_samples_df.to_pandas()

selected_samples_df_pd['enc_id'].value_counts()

################################

import matplotlib.pyplot as plt

# Convert only needed columns to numpy (cheap)
gates = selected_samples_df.select("gates").to_numpy().flatten()
depth = selected_samples_df.select("depth").to_numpy().flatten()

plt.figure(figsize=(8, 6))
plt.scatter(gates, depth, s=1, alpha=0.2)
plt.xlabel("Gates")
plt.ylabel("Depth")
plt.title("Scatter Plot of Gates vs Depth")
plt.show()
