import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tqdm import tqdm
from loguru import logger

import argparse
import os
import multiprocessing as mp
from dotenv import load_dotenv

load_dotenv()


def _process_one(args: tuple[Path, int]):
    res_dir, iter_number = args
    res_pqt_fp = res_dir / "results_db.parquet"
    if not res_pqt_fp.exists():
        return None

    dn = "".join(filter(str.isdigit, res_dir.name))

    data_record_fp = res_dir / "data_record.json"
    chain_pos = None
    mt_gates = None
    if data_record_fp.exists():
        try:
            with open(data_record_fp, "r") as f:
                data_record = json.load(f)
            chain_pos = data_record.get("chain_position", {}).get("value")
            mt_gates = data_record.get("mockturtle_gates", {}).get("value")
        except Exception:
            pass

    swact = None
    len_ = None
    swact_record = res_dir / "swact_data_record.json"
    if swact_record.exists():
        try:
            with open(swact_record, "r") as f:
                swact_data_record = json.load(f)
            swact = swact_data_record["results_dict"]["data"].get("swact_count_weighted")
            len_ = swact_data_record["results_dict"]["data"].get("n_cycles")
        except Exception:
            pass

    return {
        "design_number": dn,
        "db_len": len_,
        "chain_pos": chain_pos,
        "mt_gates": mt_gates,
        "swact": swact,
        "iter_number": iter_number,
        "res_pqt_fp": str(res_pqt_fp),
    }


def get_results(folder_path: Path, debug: bool = False, workers: int | None = None):
    # Extract iteration number from folder path
    output_diranme = folder_path.parent.name
    iter_number = output_diranme.split("_")[-1]
    iter_number = int("".join(filter(str.isdigit, iter_number)))

    if "gen_iter0" in output_diranme:
        iter_number = -1

    res_dirs = [p for p in folder_path.iterdir() if p.is_dir()]
    desc = f"designs {folder_path.parent.name}"

    if debug:
        # Sequential processing for easier debugging
        iter_res_dicts: list[dict] = []
        for res_dir in tqdm(res_dirs, desc=desc, total=len(res_dirs)):
            item = _process_one((res_dir, iter_number))
            if item is not None:
                iter_res_dicts.append(item)
        if len(iter_res_dicts) == 0:
            logger.warning(
                f"No result_db entries found in '{folder_path}'. Database appears empty for iter {iter_number}."
            )
        return iter_res_dicts

    # Parallel processing
    n_workers = workers or os.cpu_count() or 1
    with mp.Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_process_one, ((rd, iter_number) for rd in res_dirs)),
                desc=desc,
                total=len(res_dirs),
            )
        )
    filtered = [r for r in results if r is not None]
    if len(filtered) == 0:
        logger.warning(f"No result_db entries found in '{folder_path}'. Database appears empty for iter {iter_number}.")
    return filtered
    # print(f"Done: {res_dir.name} | Length: {len_} | chain_pos: {chain_pos} | mt_gates: {mt_gates} | swact: {swact}")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--experiment_dirpath",
        type=str,
        help="Path to the output directory",
        default=f"{os.environ.get('WORK_DIR')}/output/multiplier_4bi_8bo_permuti_flowy",
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in single-process mode for easier debugging",
    )
    args = arg_parser.parse_args()

    experiment_dirpath = Path(args.experiment_dirpath)

    # db_path = output_dir_path / "analysis_out/swact_analysis.db.pqt"
    # df = pd.read_parquet(db_path)
    # dns = df.sort_values(by="swact_weighted_average", ascending=True)
    folder_list = [
        "again_flowy_run_12chains_3000steps_gen_iter",
        "again_flowy_run_12chains_3000steps_proto_iter",
    ]
    # folder_list = ["flowy_run_mixed_12chains_3000steps_gen_iter", "flowy_run_mixed_12chains_3000steps_proto_iter"]
    all_res_dicts = []
    for folder in experiment_dirpath.iterdir():
        if any(f in folder.name for f in folder_list):
            if not folder.name.startswith("debug"):
                synth_out_dirpath = folder / "synth_out"
                all_res_dicts.extend(get_results(folder_path=synth_out_dirpath, debug=args.debug))

    if len(all_res_dicts) == 0:
        logger.warning(
            f"No results collected from '{experiment_dirpath}'. No matching 'results_db.parquet' were found across iterations."
        )
        return

    # Build DataFrame and plot
    all_dbs_len_df = pd.DataFrame(all_res_dicts)
    if all_dbs_len_df.empty:
        logger.warning("Aggregated results DataFrame is empty. Nothing to analyze or plot.")
        return
    all_dbs_len_df = all_dbs_len_df.query("mt_gates < 350")
    if all_dbs_len_df.empty:
        logger.warning("No rows remain after filtering on 'mt_gates < 350'. Skipping plots and summaries.")
        return
    plot_swact_vs_mt_gates(all_dbs_len_df, color_by="chain_pos")
    plot_swact_vs_mt_gates(all_dbs_len_df, color_by="iter_number")

    # Summaries
    best_designs = all_dbs_len_df.sort_values(by="swact", ascending=True).head(10)
    logger.info(f"Best designs for swact are: {best_designs}")

    best_designs = all_dbs_len_df.sort_values(by="mt_gates", ascending=True).head(10)
    logger.info(f"Best designs for mt_gates are: {best_designs}")

    # Optionally re-open parquet for best designs if needed
    for _, best_design in best_designs.iterrows():
        res_dir = Path(best_design["res_pqt_fp"]).parent
        sim_results_db_filepath = res_dir / "results_db.parquet"
        if sim_results_db_filepath.exists():
            _ = pd.read_parquet(sim_results_db_filepath)

    _ = all_dbs_len_df.query("swact < 1000 and mt_gates > 260")


def plot_swact_vs_mt_gates(df: pd.DataFrame, color_by: str):
    """
    Plots 'swact' vs 'mt_gates', colored by a third column.

    Parameters:
    - df: pandas DataFrame containing the columns.
    - color_by: name of the column to use for color mapping.
    """
    if not {"swact", "mt_gates", color_by}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'swact', 'mt_gates', and '{}'".format(color_by))

    if df.empty:
        logger.warning(f"No data available to plot (color_by='{color_by}'). Skipping figure generation.")
        return

    if color_by == "iter_number":
        cmap = plt.get_cmap("viridis_r", len(df[color_by].unique()))
        # cmap ={
        #     0:"#fde725",
        #     1:"#d8e219",
        #     2:"#b0dd2f",
        #     3:"#89d548",
        #     4:"#65cb5e",
        #     5:"#46c06f",
        #     6:"#2eb37c",
        #     7:"#1f978b",
        #     8:"#23898e",
        #     9:"#2e6d8e",
        #     10:"#355e8d",
        #     11:"#3d4e8a",
        #     12:"#433d84",
        #     13:"#472a7a",
        #     14:"#9C4444",
        #     15:"#97448C",
        #     16:"#97448C",
        # }
        # df["color"] = (df[color_by] + 1).map(cmap)
    else:
        cmap = "viridis"

    great_designs = {
        # "sme_lut": {"mt_gates": 73, "swact": 246.59, "chain_pos": 1984, "c": "darkslategrey", "name": "SME"},
        "sme_lut": {"mt_gates": 69, "swact": 233, "chain_pos": 1984, "c": "darkslategrey", "name": "SME"},
        # "sme_handmade": {"mt_gates": 67, "swact": 248.37, "chain_pos": 2, "c": "darkslategrey", "name": "SME"},
        # "tc_lut": {"mt_gates": 83, "swact": 386.81, "chain_pos": 696, "c": "teal", "name": "SME"},
        "tc_lut": {"mt_gates": 64, "swact": 402, "chain_pos": 696, "c": "teal", "name": "TC"},
        # "tc_handmade": {"mt_gates": 68, "swact": 362.43, "chain_pos": 2, "c": "teal", "name": "TC"},
    }

    gen0_df = df[df["iter_number"] == -1]
    added_dataset = gen0_df[gen0_df["design_number"].apply(int) >= 10000]
    df = df[df["design_number"].apply(int) <= 10000]

    plt.rcParams["font.size"] = 10
    fig, ax = plt.subplots(figsize=(8, 3))
    if True:
        scatter = plt.scatter(df["mt_gates"], df["swact"], c=df[color_by], cmap=cmap, s=1.0, alpha=1.0, rasterized=True)
    else:
        scatter = plt.scatter(
            df["mt_gates"], df["swact"], c=list(df["color"].values), s=1.0, alpha=1.0, rasterized=True
        )
    plt.scatter(
        added_dataset["mt_gates"], added_dataset["swact"], c="orange", marker="x", s=1.0, alpha=1.0, rasterized=True
    )
    for label in ["sme_lut", "tc_lut"]:
        ax.scatter(
            great_designs[label]["mt_gates"],
            great_designs[label]["swact"],
            c="deeppink",
            marker="*",
            s=1.0,
            alpha=1.0,
            rasterized=True,
        )
        # ax.text(great_designs[label]["mt_gates"], great_designs[label]["swact"], great_designs[label]["name"], fontsize=7, color="deeppink", ha="left", va="bottom")
    plt.colorbar(scatter, label="Generation Round")
    plt.xlabel("MIG Gate Count (NMIG)")
    plt.ylabel("Mean SwAct (a.u.)")
    # plt.title(f"swact vs mt_gates (colored by {color_by}) | nb_designs={len(df)}")
    # plt.title(f"swact vs mt_gates (colored by {color_by}) | nb_designs={len(df)}")
    plt.minorticks_on()
    plt.grid(visible=True, which="major", color="grey", linewidth=0.5, linestyle="-", alpha=0.5)
    plt.grid(visible=True, which="minor", color="grey", linewidth=0.5, linestyle="--", alpha=0.2)
    ax.tick_params(axis="y", labelrotation=70)
    plt.setp(ax.get_yticklabels(), va="center")
    plt.tight_layout()

    # -- Create inset axes in the bottom right
    # [x0, y0, width, height] in axes fraction (0-1)
    inset_ax = inset_axes(ax, width="45%", height="45%", loc="lower right")

    # -- Zoom in on bottom left corner of main plot
    xlim_inset = ax.get_xlim()
    xlim_inset = (60, 100)
    ylim_inset = ax.get_ylim()
    ylim_inset = (220, 450)
    if True:
        inset_ax.scatter(df["mt_gates"], df["swact"], c=df[color_by] + 1, cmap=cmap, s=1.0, alpha=1.0, rasterized=True)
    else:
        inset_ax.scatter(df["mt_gates"], df["swact"], c=list(df["color"].values), s=1.0, alpha=1.0, rasterized=True)
    inset_ax.scatter(
        added_dataset["mt_gates"], added_dataset["swact"], c="orange", marker="x", s=1.0, alpha=1.0, rasterized=True
    )
    for label in ["sme_lut", "tc_lut"]:
        inset_ax.scatter(
            great_designs[label]["mt_gates"],
            great_designs[label]["swact"],
            c="deeppink",
            marker="*",
            s=1.0,
            alpha=1.0,
            rasterized=True,
        )
        inset_ax.text(
            great_designs[label]["mt_gates"],
            great_designs[label]["swact"],
            great_designs[label]["name"],
            fontsize=7,
            color="deeppink",
            ha="left",
            va="bottom",
        )

    for ticklabel in inset_ax.get_yticklabels():
        ticklabel.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="none"))

    inset_ax.set_xlim(xlim_inset)
    inset_ax.set_ylim(ylim_inset)
    inset_ax.minorticks_on()
    inset_ax.grid(visible=True, which="major", color="grey", linewidth=0.5, linestyle="-", alpha=0.5)
    inset_ax.grid(visible=True, which="minor", color="grey", linewidth=0.5, linestyle="--", alpha=0.2)
    # inset_ax.tick_params(axis='y', labelrotation=90)
    plt.setp(inset_ax.get_yticklabels(), va="center")

    # Optional: highlight the region on the main plot
    rect = plt.Rectangle(
        (xlim_inset[0], ylim_inset[0]),
        xlim_inset[1] - xlim_inset[0],
        ylim_inset[1] - ylim_inset[0],
        linewidth=0.5,
        edgecolor="deeppink",
        facecolor="none",
    )
    ax.add_patch(rect)

    plt.savefig(f"swact_vs_mt_gates_cby{color_by}.png", dpi=300)
    plt.savefig(f"swact_vs_mt_gates_cby{color_by}.pdf", dpi=300)
    logger.info(f"Done plotting swact vs mt_gates, save in: swact_vs_mt_gates_cby{color_by}.png")


if __name__ == "__main__":
    main()
