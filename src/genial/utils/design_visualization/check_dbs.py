import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt

from tqdm import tqdm
from loguru import logger

import argparse
import os

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--output_dirpath",
    type=str,
    help="Path to the output directory",
    default=f"{os.environ.get('WORK_DIR')}/output/multiplier_4bi_8bo_notech_normal_only/loop_synth_gen_iter0",
)
args = arg_parser.parse_args()

output_dir_path = Path(args.output_dirpath)

# db_path = output_dir_path / "analysis_out/swact_analysis.db.pqt"
# df = pd.read_parquet(db_path)
# dns = df.sort_values(by="swact_weighted_average", ascending=True)

synth_out_dirpath = output_dir_path / "synth_out"

all_res_dicts = []
for res_dir in tqdm(synth_out_dirpath.iterdir(), desc="designs", total=len(list(synth_out_dirpath.iterdir()))):
    res_pqt_fp = res_dir / "results_db.parquet"
    if res_pqt_fp.exists():
        # rs_df = pd.read_parquet(res_pqt_fp)

        dn = "".join(filter(str.isdigit, res_dir.name))

        data_record_fp = res_pqt_fp = res_dir / "data_record.json"
        if data_record_fp.exists():
            with open(data_record_fp, "r") as f:
                data_record = json.load(f)

            chain_pos = data_record["chain_position"]["value"]
            mt_gates = data_record["mockturtle_gates"]["value"]
        else:
            chain_pos = None
            mt_gates = None

        swact_record = res_dir / "swact_data_record.json"
        if swact_record.exists():
            with open(swact_record, "r") as f:
                swact_data_record = json.load(f)
            swact = swact_data_record["results_dict"]["data"]["swact_count_weighted"]
            len_ = swact_data_record["results_dict"]["data"]["n_cycles"]
        else:
            swact = None
            len_ = None

        all_res_dicts.append(
            {
                "design_number": dn,
                "db_len": len_,
                "chain_pos": chain_pos,
                "mt_gates": mt_gates,
                "swact": swact,
            }
        )
        # print(f"Done: {res_dir.name} | Length: {len_} | chain_pos: {chain_pos} | mt_gates: {mt_gates} | swact: {swact}")


def plot_swact_vs_mt_gates(
    df: pd.DataFrame,
    color_by: str,
    output_dir: Path,
    special_designs_path: Path | None = None,
):
    """
    Plots 'swact' vs 'mt_gates', colored by a third column.

    Parameters:
    - df: pandas DataFrame containing the columns.
    - color_by: name of the column to use for color mapping.
    """
    if not {"swact", "mt_gates", color_by}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'swact', 'mt_gates', and '{}'".format(color_by))

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        df["mt_gates"],
        df["swact"],
        c=df[color_by],
        cmap="viridis",
        edgecolor="k",
        alpha=0.8,
    )
    plt.colorbar(scatter, label=color_by)
    plt.xlabel("mt_gates")
    plt.ylabel("swact")
    plt.title(f"swact vs mt_gates (colored by {color_by}) | nb_designs={len(df)}")
    # Add labels for special designs, if provided
    if special_designs_path is not None and special_designs_path.exists():
        try:
            with open(special_designs_path, "r") as f:
                sd = json.load(f)

            legends = sd.get("legend", [])
            design_numbers = sd.get("design_numbers", [])
            if len(legends) != len(design_numbers):
                logger.warning(
                    "special_designs.json has mismatched lengths: legend={}, design_numbers={}",
                    len(legends),
                    len(design_numbers),
                )
            # Map design_number -> label (zip truncates to shortest)
            dn_to_label = dict(zip(design_numbers, legends))

            # Annotate matching points with improved placement
            if "design_number" in df.columns:
                ax = plt.gca()
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()

                # Filter to special designs present in df with valid coords
                mask = df["design_number"].isin(dn_to_label.keys()) & df["mt_gates"].notna() & df["swact"].notna()
                sdf = df.loc[mask, ["design_number", "mt_gates", "swact"]].copy()

                # Bucket by x to separate labels that are close horizontally
                # Use a bucket size relative to the x-range
                xr = max(x_max - x_min, 1e-9)
                bucket_size = xr * 0.01  # 1% of the x-range
                sdf["_xbucket"] = ((sdf["mt_gates"] - x_min) / bucket_size).round().astype(int)

                # Prepare alternating vertical offsets within each bucket
                bucket_counts: dict[int, int] = {}

                for _, row in sdf.iterrows():
                    dn = row["design_number"]
                    x = float(row["mt_gates"])  # data coords
                    y = float(row["swact"])  # data coords

                    # Edge-aware preferred directions
                    x_pos = (x - x_min) / xr
                    yr = max(y_max - y_min, 1e-9)
                    y_pos = (y - y_min) / yr

                    prefer_left = x_pos > 0.85  # near right edge -> place label on left
                    prefer_right = x_pos < 0.15
                    prefer_above = y_pos < 0.8  # if not near top, place above
                    prefer_below = y_pos > 0.9

                    # Alternating vertical offsets per bucket to reduce overlap
                    b = int(row["_xbucket"])
                    idx_in_bucket = bucket_counts.get(b, 0)
                    bucket_counts[b] = idx_in_bucket + 1

                    # Cycle through a set of base offsets (in points)
                    base_offsets = [(10, 10), (10, -10), (-10, 10), (-10, -10), (14, 0), (0, 14), (-14, 0), (0, -14)]
                    dx_pts, dy_pts = base_offsets[idx_in_bucket % len(base_offsets)]

                    # Apply edge preferences
                    if prefer_left:
                        dx_pts = -abs(dx_pts)
                    elif prefer_right:
                        dx_pts = abs(dx_pts)

                    if prefer_below:
                        dy_pts = -abs(dy_pts) if dy_pts != 0 else -10
                    elif prefer_above:
                        dy_pts = abs(dy_pts) if dy_pts != 0 else 10

                    # Alignment according to offset
                    ha = "left" if dx_pts >= 0 else "right"
                    va = "bottom" if dy_pts >= 0 else "top"

                    ax.annotate(
                        dn_to_label[dn],
                        (x, y),
                        textcoords="offset points",
                        xytext=(dx_pts, dy_pts),
                        ha=ha,
                        va=va,
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"),
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="black", alpha=0.6),
                        clip_on=False,
                    )
            else:
                logger.warning("DataFrame has no 'design_number' column; cannot label special designs.")
        except Exception as e:
            logger.exception(f"Failed to process special_designs.json: {e}")

    plt.tight_layout()
    out_fp = output_dir / "swact_vs_mt_gates.png"
    plt.savefig(str(out_fp), dpi=250)


all_dbs_len_df = pd.DataFrame(all_res_dicts)

special_designs_json = output_dir_path / "special_designs.json"
plot_swact_vs_mt_gates(
    all_dbs_len_df,
    color_by="chain_pos",
    output_dir=output_dir_path,
    special_designs_path=special_designs_json if special_designs_json.exists() else None,
)
logger.info(f"Done plotting swact vs mt_gates, saved in: {(output_dir_path / 'swact_vs_mt_gates.png')}")

best_designs = all_dbs_len_df.sort_values(by="swact", ascending=True).head(10)
logger.info(f"Best designs are: {best_designs}")

for idx, best_design in best_designs.iterrows():
    dn = best_design["design_number"]
    best_design_dirpath = synth_out_dirpath / f"res_{dn}"

    sim_results_db_filepath = best_design_dirpath / "results_db.parquet"

    if sim_results_db_filepath.exists():
        best_df = pd.read_parquet(sim_results_db_filepath)

selected_designs = all_dbs_len_df.query("swact < 1000 and mt_gates > 260")
