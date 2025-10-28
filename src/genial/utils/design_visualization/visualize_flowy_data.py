import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from loguru import logger
import os

# argparser = argparse.ArgumentParser()
# argparser.add_argument("--design_number", "-dn", help="design number", type=str)
# args = argparser.parse_args()

swact_db_path = f"{os.environ.get('WORK_DIR')}/output/multiplier_4bi_8bo_mixed_permuti_flowy/flowy_loop_gen_iter0/analysis_out/swact_analysis.db.pqt"
swact_df = pd.read_parquet(swact_db_path)

dns = swact_df.sort_values(by="swact_weighted_average", ascending=True)

synth_out_dirpath = Path(
    f"{os.environ.get('WORK_DIR')}/output/multiplier_4bi_8bo_mixed_permuti_flowy/flowy_loop_gen_iter0/synth_out"
)


all_dicts = []
best_dicts = []
problems = 0
problems_dns = []
for res_dir in tqdm(synth_out_dirpath.iterdir(), desc="designs", total=len(list(synth_out_dirpath.iterdir()))):
    design_number = "".join(filter(str.isdigit, res_dir.name))

    flowy_db = res_dir / "flowy_data_record.parquet"
    if flowy_db.exists():
        flowy_df = pd.read_parquet(flowy_db)
        if flowy_df.empty:
            logger.info(f"flowy_data_record.parquet is empty for {design_number}")
            continue
    else:
        continue
        # raise FileNotFoundError(f"flowy_data_record.parquet not found in {design_synth_res_dir_path}")

        # fig = plt.figure(figsize=(7, 5))
        # for run_id in flowy_df["run_identifier"].unique():
        #     run_df = flowy_df[flowy_df["run_identifier"] == run_id]
        #     scatter = plt.scatter(
        #         run_df['step'], run_df['mockturtle_gates'],
        #         cmap='viridis', alpha=0.5, label=run_id, s=10
        #     )
        # plt.legend()
        # plt.xlabel('step')
        # plt.ylabel('mt_gates')
        # plt.title(f"Nb runs: {len(flowy_df['run_identifier'].unique())}")
        # plt.tight_layout()
        # plt.savefig(f"zzz_dn{args.design_number}_mt_gates_vs_step.png", dpi=250)
        # plt.close()

    run_dicts = []
    min_swact = float("inf")
    min_mt_gates = float("inf")

    for run_id in flowy_df["run_identifier"].unique():
        run_df = flowy_df[flowy_df["run_identifier"] == run_id]
        run_dicts.append(
            {
                "swact": run_df["swact_count_weighted"].iloc[0],
                "best_mt_gates": run_df["mockturtle_gates"].sort_values(ascending=True).iloc[0],
                "mean_mt_gates": run_df["mockturtle_gates"].mean(),
                "design_number": design_number,
            }
        )
        try:
            min_swact = min(min_swact, run_dicts[-1]["swact"])
            min_mt_gates = min(min_mt_gates, run_dicts[-1]["best_mt_gates"])
        except Exception:
            logger.info(run_dicts[-1])
            problems += 1
            problems_dns.append(design_number)
            break

    if len(run_dicts) > 2:
        run_df = pd.DataFrame(run_dicts)
        pearson = run_df["best_mt_gates"].corr(run_df["swact"], method="pearson")
        run_best_dict = {
            "min_swact": min_swact,
            "min_mt_gates": min_mt_gates,
            "pearson": pearson,
            "design_number": design_number,
        }
        best_dicts.append(run_best_dict)

    all_dicts.extend(run_dicts)

plot_df = pd.DataFrame(all_dicts)

logger.info(f"Plotting swact vs mt_gates for {len(plot_df['design_number'].unique())} designs")
fig = plt.figure(figsize=(7, 5))
scatter = plt.scatter(plot_df["best_mt_gates"], plot_df["swact"], cmap="viridis", alpha=0.5)
plt.xlabel("best_mt_gates")
plt.ylabel("swact")
plt.title(f"Nb designs: {len(plot_df['design_number'].unique())} | Nb points: {len(plot_df)}")
plt.tight_layout()
plt.savefig(f"zzz_all_swact_vs_best_mtgates.png", dpi=250)
plt.close()

logger.info(f"Saved at: {Path(f'zzz_all_swact_vs_best_mtgates.png')}")

logger.info(f"Plotting swact vs mt_gates for {len(plot_df['design_number'].unique())} designs")
fig = plt.figure(figsize=(7, 5))
scatter = plt.scatter(plot_df["mean_mt_gates"], plot_df["swact"], cmap="viridis", alpha=0.5)
plt.xlabel("mean_mt_gates")
plt.ylabel("swact")
plt.title(f"Nb designs: {len(plot_df['design_number'].unique())} | Nb points: {len(plot_df)}")
plt.tight_layout()
plt.savefig(f"zzz_all_swact_vs_mean_mtgates.png", dpi=250)
plt.close()

logger.info(f"Saved at: {Path(f'zzz_all_swact_vs_mean_mtgates.png')}")


pearson_plot_df = pd.DataFrame(best_dicts)

logger.info(f"Plotting pearson vs mt_gates for {len(pearson_plot_df['design_number'].unique())} designs")
fig = plt.figure(figsize=(7, 5))
scatter = plt.scatter(
    pearson_plot_df["min_mt_gates"],
    pearson_plot_df["pearson"],
    c=pearson_plot_df["min_swact"],
    cmap="viridis",
    alpha=0.5,
)
plt.colorbar(scatter, label="min_swact")
plt.xlabel("min_mt_gates")
plt.ylabel("per design pearson coefficient")
plt.title(f"Nb designs: {len(pearson_plot_df['design_number'].unique())} | Nb points: {len(pearson_plot_df)}")
plt.tight_layout()
plt.savefig(f"zzz_all_pearson_vs_min_mtgates.png", dpi=250)
plt.close()


# def plot_swact_vs_mt_gates(df: pd.DataFrame, color_by: str):
#     """
#     Plots 'swact' vs 'mt_gates', colored by a third column.

#     Parameters:
#     - df: pandas DataFrame containing the columns.
#     - color_by: name of the column to use for color mapping.
#     """
#     if not {'swact', 'mt_gates', color_by}.issubset(df.columns):
#         raise ValueError("DataFrame must contain 'swact', 'mt_gates', and '{}'".format(color_by))

#     plt.figure(figsize=(7, 5))
#     scatter = plt.scatter(
#         df['mt_gates'], df['swact'],
#         c=df[color_by], cmap='viridis', edgecolor='k', alpha=0.8
#     )
#     plt.colorbar(scatter, label=color_by)
#     plt.xlabel('mt_gates')
#     plt.ylabel('swact')
#     plt.title(f'swact vs mt_gates (colored by {color_by})')
#     plt.tight_layout()
#     plt.savefig("swact_vs_mt_gates.png", dpi=250)


# # all_dbs_len_df = pd.DataFrame(all_res_dicts)
# pl/ot_swact_vs_mt_gates(all_dbs_len_df, color_by="chain_pos")
