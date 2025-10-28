# env variable WORK_DIR
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from genial.config.config_dir import ConfigDir
from genial.experiment.task_analyzer import Analyzer
from genial.training.elements.score_tools import ScoreComputeHelper

import genial.experiment.plotter as plotter

from loguru import logger
import traceback

import argparse
from typing import Any


def get_dir_name(prefix: str, desc: str, generation: int) -> str:
    """
    Generate a directory name with the given prefix, description, and generation number.
    """
    assert prefix is not None
    assert generation is not None
    return f"{prefix}_{desc}_iter{generation}"


def setup_output_dir_name(dir_name_base: str, desc: str, output_dir_name: str | None = None, iteration: int = 0):
    if output_dir_name is not None:
        _output_dir_name = output_dir_name
    else:
        _output_dir_name = get_dir_name(prefix=dir_name_base, desc=desc, generation=iteration)

    return _output_dir_name


def setup_analyzer(steps: list[str], experiment_name: str, output_dir_name: str):
    args_dict = {
        "experiment_name": experiment_name,
        "output_dir_name": output_dir_name,
    }

    if "synth" not in steps:
        args_dict["skip_synth"] = True

    if "swact" not in steps:
        args_dict["skip_swact"] = True

    if "power" not in steps:
        args_dict["skip_power"] = True
    else:
        args_dict["bulk_flow_dirname"] = "power_out"
        args_dict["technology"] = "asap7"

    dir_config = ConfigDir(is_analysis=True, **args_dict)

    analyzer = Analyzer(
        dir_config=dir_config,
        read_only=True,
    )

    return analyzer


def format_eng(value):
    print(value)
    if value >= 1e9:
        return f"{value / 1e9:.2f}G"
    elif value >= 1e6:
        return f"{value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"{value / 1e3:.2f}k"
    elif value >= 1e0:
        return f"{value:.2f}"
    elif value >= 1e-3:
        return f"{value * 1e3:.2f}m"
    elif value >= 1e-6:
        return f"{value * 1e6:.2f}u"
    elif value >= 1e-9:
        return f"{value * 1e9:.2f}n"
    elif value >= 1e-12:
        return f"{value * 1e12:.2f}p"
    else:
        return f"{value:.2f}"


def get_stat(
    steps: list[str],
    keys: list[str],
    experiment_name: str = None,
    dir_name_base: str = None,
    desc: str = None,
    output_dir_name: str | None = None,
    iteration=0,
) -> tuple[pd.DataFrame, Path | None]:
    try:
        _output_dir_name = setup_output_dir_name(
            dir_name_base=dir_name_base,
            desc=desc,
            output_dir_name=output_dir_name,
            iteration=iteration,
        )
        analyzer = setup_analyzer(
            steps=steps,
            experiment_name=experiment_name,
            output_dir_name=_output_dir_name,
        )

        analyzer.align_databases()

        dfs = {step: getattr(analyzer, f"{step}_df") for step in steps}

        # Merge all dfs together:
        if len(dfs) > 1:
            logger.info(f"Merging all databases for steps: {steps}")
            merged_df = None
            for index in range(len(dfs) - 1):
                step = steps[index]
                step_next = steps[index + 1]
                logger.info(step)
                try:
                    if index == 0:
                        merged_df = ScoreComputeHelper.merge_data_df(dfs[step], dfs[step_next], suffix=f"_{step}")
                    else:
                        merged_df = ScoreComputeHelper.merge_data_df(merged_df, dfs[step_next], suffix=f"_{step}")
                except Exception as e:
                    logger.error(f"Encountered an issue while trying to merge step {step} with {step_next}")
                    raise e
        else:
            merged_df = dfs[0]

        df_stat = dict()

        logger.info(f"=========================================================== desc: {desc} | iter {iteration}")

        for key in keys:
            logger.info(f"===== key: {key}")

            # logger.info(f"Sorting by {key}")
            # logger.info(merged_df.sort_values(by=key))
            best_df = merged_df.sort_values(by=key).iloc[:5]
            print(best_df)
            best_df["iteration"] = iteration
            _stats = {
                f"{key}_min": merged_df[key].min(),
                f"{key}_mean": merged_df[key].mean(),
                f"{key}_max": merged_df[key].max(),
                f"{key}_std": merged_df[key].std(),
                f"{key}_nb": len(merged_df[key]),
                f"in_bitwidth": int(analyzer.exp_config["input_bitwidth"]),
            }
            logger.info(_stats)
            df_stat.update(_stats)
            df_stat[f"{key}_best_designs"] = best_df
        df_stat[f"iteration"] = iteration
        # df_stat["nb_designs"] = len(merged_df)
        logger.info(f"Statistics for {_output_dir_name} {iteration} {desc}: done.")

        # Check duplicates only on the 'encodings_input' column
        df_without_duplicates = merged_df.drop_duplicates(["encodings_input"])
        n_duplicates = len(merged_df) - len(df_without_duplicates)
        df_stat["n_duplicates"] = n_duplicates

        if iteration == 0 and (output_dir_name is not None or desc == "gen"):
            return df_stat, analyzer.plot_dir.parent
        else:
            return df_stat, None

    except FileExistsError as e:
        logger.info(f"File not found: {e}: {_output_dir_name}")
        return_dict = {}
        for key in keys:
            return_dict.update(
                {
                    f"{key}_min": np.nan,
                    f"{key}_mean": np.nan,
                    f"{key}_max": np.nan,
                    f"{key}_std": np.nan,
                    f"{key}_nb": np.nan,
                    f"{key}_best_designs": pd.DataFrame(),
                }
            )
        return_dict["n_duplicates"] = np.nan
        return return_dict, None


def get_lists(
    steps: list[str],
    keys: list[str],
    n_iter_max: int,
    experiment_name: str,
    dir_name_base: str,
    origin_output_dir_name: str | None = None,
):
    full_set = []
    proto_set = []
    origin_output_dir_path = None

    for i in range(n_iter_max):
        try:
            if i == 0:
                if origin_output_dir_name is not None:
                    v, origin_output_dir_path = get_stat(
                        steps=steps,
                        keys=keys,
                        experiment_name=experiment_name,
                        output_dir_name=origin_output_dir_name,
                        desc="gen",
                        iteration=i,
                    )
                else:
                    v, origin_output_dir_path = get_stat(
                        steps=steps,
                        keys=keys,
                        experiment_name=experiment_name,
                        dir_name_base=dir_name_base,
                        desc="gen",
                        iteration=i,
                    )
                full_set.append(v)
            v, _ = get_stat(
                steps=steps,
                keys=keys,
                experiment_name=experiment_name,
                dir_name_base=dir_name_base,
                desc="proto",
                iteration=i,
            )
            proto_set.append(v)
            v, _ = get_stat(
                steps=steps,
                keys=keys,
                experiment_name=experiment_name,
                dir_name_base=dir_name_base,
                desc="merge",
                iteration=i,
            )
            full_set.append(v)
        except Exception:
            logger.info(f"Iteraiton {i} is probably not done yet.")
            logger.info(traceback.format_exc())

    return full_set, proto_set, origin_output_dir_path


def do_plots_old(full_set, proto_set, key, label, origin_output_dir_path):
    colors = ["blue", "orange", "green", "red", "purple"]
    fullset_style = "solid"
    protoset_style = "dashed"

    subkeys = ["mean", "min"]

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))
    for idx, ax in enumerate(axes):
        subkey = subkeys[idx]

        # extract the series
        print()
        full = np.array([v[f"{key}_{subkey}"] for v in full_set])
        proto = np.array([v[f"{key}_{subkey}"] for v in proto_set])
        logger.info(f"{key}_{subkey} ---- label {label} --- value full[0]={full[0]}")
        proto = np.append(full[0], proto)
        logger.info(f"{key}_{subkey} ---- label {label} --- value proto[0]={proto[0]}")

        # plot means
        # ax.plot(full, linestyle=fullset_style, color=colors[0], label=f"{label} full")
        ax.plot(proto, linestyle=protoset_style, color=colors[0], label=f"{label} proto")

        # reitck the x axis, starting from -1 instead of 0
        ax.set_xticks(np.arange(len(proto)))
        ax.set_xticklabels(np.arange(-1, len(proto) - 1))

        # annotate the means subplot
        ax.set_ylabel(f"{key}")
        ax.set_title(f"{subkey} {key} per Iteration")
        ax.legend(loc="best")
        ax.grid(True)

        if idx == 1:
            ax.set_xlabel("Iteration")

        # Duplicate y axises to plot relative in relative scale
        ax1 = ax.twinx()
        ax1.plot(
            proto / proto[0], linestyle=protoset_style, color=colors[1], label=f"{label} proto relative", alpha=0.0
        )
        ax1.set_ylabel(f"(relative)", color=colors[1])
        ax1.grid(visible=True, which="major", color=colors[1], linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(origin_output_dir_path / "plots_loop_analysis" / f"{key}_loop.png")
    logger.info(f"Plot saved as {origin_output_dir_path}/plots_loop_analysis/{key}_loop.png")

    # new plot for duplicates
    fig, ax = plt.subplots(figsize=(10, 6))
    # extract the series
    n_duplicates_full = [v["n_duplicates"] for v in full_set]
    n_duplicates_proto = [v["n_duplicates"] for v in proto_set]

    # plot means
    ax.plot(n_duplicates_full, linestyle=fullset_style, color=colors[0], label=f"{label} full")
    ax.plot(n_duplicates_proto, linestyle=protoset_style, color=colors[0], label=f"{label} proto")
    # annotate the duplicates subplot
    ax.set_xlabel("Iteration")
    ax.set_ylabel("# Duplicates")
    ax.set_title("Number of Duplicates per Iteration")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(origin_output_dir_path / "plots_loop_analysis" / "duplicates_loop.png")
    logger.info(f"Plot saved as {origin_output_dir_path}/plots_loop_analysis/duplicates_loop.png")

    return None


axis_key_name_map = {
    "p_comb_dynamic": "Dynamic Power (W)",
    "complexity_post_opt": "Complexity (a.u.)",
    "nb_transistors": "# Transistors",
    "swact_weighted_average": "Mean SwAct (a.u.)",
    "mockturtle_gates": "#MIG",
}
axis_subkey_name_map = {
    "min": "Minimum",
    "mean": "Mean",
}


def do_plots(
    full_set,
    proto_set,
    key,
    label,
    origin_output_dir_path,
    sme_value: float | None = None,
    tc_value: float | None = None,
):
    # fullset_style = "solid"
    # protoset_style = "solid"

    # sme_value/tc_value now provided per-run via runs_dict

    subkeys = ["mean", "min"]
    styles = ["dashed", "solid"]

    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(10, 3))
    # ax1 = ax0
    # ax1 = ax0.twinx()
    # axes = [ax0, ax1]
    for idx in range(1, -1, -1):
        ax = axes[1 - idx]
        subkey = subkeys[idx]

        # extract the series
        full = np.array([v[f"{key}_{subkey}"] for v in full_set])
        proto = np.array([v[f"{key}_{subkey}"] for v in proto_set])
        # print(list(v.keys() for v in proto_set))
        logger.info(f"{key}_{subkey} ---- label {label} --- value full[0]={full[0]}")
        proto = np.append(full[0], proto)
        logger.info(f"{key}_{subkey} ---- label {label} --- value proto[0]={proto[0]}")
        nbs = np.array([v[f"{key}_nb"] for v in proto_set])
        nbs = np.append(full_set[0][f"{key}_nb"], nbs)

        proto_arr = np.array(proto)
        # remove all nan values from proto
        proto_arr = proto_arr[~np.isnan(proto_arr)]

        # if idx == 0:
        #     ax1 = ax.twinx()
        #     ax1.bar(
        #         x=range(len(proto_arr)),
        #         height=nbs[: len(proto_arr)],
        #         color="gray",
        #         alpha=0.5,
        #         # label="nb",
        #         width=0.5,
        #     )
        #     ax1.set_ylabel("# Designs (bars)")
        #     ax1.grid(True, which="major", linestyle="-")

        # plot means
        # ax.plot(full, linestyle=fullset_style, color=colors[0], label=f"{label} full")
        # ax.plot(proto, linestyle=protoset_style, color=colors[0], label=f"{label} proto")
        # if do_color_numbers:
        ax.plot(
            range(len(proto)),
            proto,
            marker="^",
            linestyle=styles[idx],
            linewidth=1.0,
            color="#1f77b4",
            label=f"{axis_subkey_name_map[subkey]}",
            zorder=0,
        )
        if 1 - idx == 0:
            scatter = ax.scatter(
                range(len(proto)),
                proto,
                marker="^",
                # linestyle=styles[idx],
                # linewidth=1.0,
                c=nbs.tolist(),
                zorder=1,
                # label=f"{axis_subkey_name_map[subkey]}",
            )
            plt.colorbar(
                scatter,
                label="Number of Designs",
                location="right",
                # shrink=0.5,
                pad=0.05,
            )
        # Add text labels at each point
        # if idx == 1:
        #     for xi, yi in zip(list(range(len(proto))), proto):
        #         ax.text(
        #             xi, yi, f"{format_eng(yi)}", ha="center", va="top", rotation=45
        #         )  # You can format as you like
        # line, label = ax.get_legend_handles_labels()
        # lines.append(line)
        # labels.append(label)
        # reitck the x axis, starting from -1 instead of 0
        ax.set_xticks(np.arange(len(proto_arr)))
        ax.set_xticklabels(np.arange(len(proto_arr)))
        if key == "swact_weighted_average" and subkey == "min" and sme_value is not None and tc_value is not None:
            # Add lines for the sme and tc values
            ax.hlines(
                [sme_value, tc_value],
                xmin=0,
                xmax=len(proto_arr) - 1,
                colors=["darkslategrey", "teal"],
                linestyles=["dashed", "dashed"],
                linewidth=1.0,
                # label=["SME", "TC"],
            )
            ax.text(1, sme_value, "SME", color="darkslategrey", ha="right", va="bottom")
            ax.text(1, tc_value, "TC", color="teal", ha="right", va="bottom")
        # ax.tick_params(axis="both", labelrotation=80)

        # annotate the means subplot
        ax.legend(loc="upper right", fontsize=8)
        if idx == 1:
            ax.set_ylabel(f"{axis_key_name_map[key]}")
        # ax.grid(True)
        ax.minorticks_on()
        ax.grid(visible=True, which="minor", color="grey", linewidth=0.5, linestyle="--", alpha=0.2)
        ax.grid(visible=True, which="major", color="grey", linewidth=0.5, linestyle="-", alpha=0.5)
        ax.set_xlabel("Generation Round")

    # axes[0].legend(lines[0] + lines[1], labels[0] + labels[1], loc="best")  # or another loc
    # plt.legend(loc="best")
    # plt.tight_layout()

    # if idx == 1:
    # ax.set_xlabel("Iteration")

    # Duplicate y axises to plot relative in relative scale
    # ax1 = ax.twinx()
    # ax1.plot(
    #     proto / proto[0], linestyle=protoset_style, color=colors[1], label=f"{label} proto relative", alpha=0.0
    # )
    # ax1.set_ylabel(f"(relative)", color=colors[1])
    # ax1.grid(visible=True, which="major", color=colors[1], linestyle="--", alpha=0.5)

    plt.tight_layout()
    if not (origin_output_dir_path / "plots_loop_analysis").exists():
        (origin_output_dir_path / "plots_loop_analysis").mkdir(parents=True)
    plt.savefig(origin_output_dir_path / "plots_loop_analysis" / f"{key}_loop.png", dpi=300)
    plt.savefig(origin_output_dir_path / "plots_loop_analysis" / f"{key}_loop.pdf")
    logger.info(f"PNG Plot saved as {origin_output_dir_path}/plots_loop_analysis/{key}_loop.png")
    logger.info(f"PDF Plot saved as {origin_output_dir_path}/plots_loop_analysis/{key}_loop.pdf")

    # new plot for duplicates
    # fig, ax = plt.subplots(figsize=(10, 6))
    # # extract the series
    # n_duplicates_full = [v["n_duplicates"] for v in full_set]
    # n_duplicates_proto = [v["n_duplicates"] for v in proto_set]

    # # plot means
    # ax.plot(n_duplicates_full, linestyle=fullset_style, color=colors[0], label=f"{label} full")
    # ax.plot(n_duplicates_proto, linestyle=protoset_style, color=colors[0], label=f"{label} proto")
    # # annotate the duplicates subplot
    # ax.set_xlabel("Iteration")
    # ax.set_ylabel("# Duplicates")
    # ax.set_title("Number of Duplicates per Iteration")
    # ax.legend(loc="best")
    # ax.grid(True)
    # plt.tight_layout()
    # plt.savefig(origin_output_dir_path / "plots_loop_analysis" / "duplicates_loop.png")
    # logger.info(f"Plot saved as {origin_output_dir_path}/plots_loop_analysis/duplicates_loop.png")

    return None


# ------------------------------------------------------------------
# Centralized run configurations
# ------------------------------------------------------------------
runs_dict: dict[str, dict[str, Any]] = {
    "again_flowy": {
        "experiment_name": "multiplier_4bi_8bo_permuti_flowy",
        "dir_name_base": "again_flowy_run_12chains_3000steps",
        "origin_output_dir_name": "again_flowy_run_12chains_3000steps_gen_iter0",
        "label": "Flowy Loop (Again) | 12chainsx3000steps | 10kinit-4kgr",
        # Reference lines for swact plots (Default)
        "sme_value": 233.4,
        "tc_value": 402.0,
        # Defaults for analysis (overridable via CLI)
        "keys": ["swact_weighted_average"],
        "steps": ["cmplx", "gener", "synth", "swact"],
    },
    "mixed_flowy": {
        "experiment_name": "multiplier_4bi_8bo_mixed_permuti_flowy",
        "dir_name_base": "flowy_run_mixed_12chains_3000steps",
        "origin_output_dir_name": "flowy_run_mixed_12chains_3000steps_gen_iter0",
        "label": "Flowy Loop (Mixed) | 12chainsx3000steps | 10kinit-4kgr",
        # Reference lines for swact plots (Mixed)
        "sme_value": 450.02,
        "tc_value": 402.0,
        # Defaults for analysis (overridable via CLI)
        "keys": ["swact_weighted_average"],
        "steps": ["cmplx", "gener", "synth", "swact"],
    },
    "bimodal_flowy": {
        "experiment_name": "multiplier_4bi_8bo_permuti_flowy_bimodal",
        "dir_name_base": "flowy_run_bimodal_12chains_3000steps",
        "origin_output_dir_name": "flowy_run_bimodal_12chains_3000steps_gen_iter0",
        "label": "Flowy Loop (Bimodal) | 12chainsx3000steps | 10kinit-4kgr",
        # Reference lines for swact plots (Bimodal)
        "sme_value": 314.66,
        "tc_value": 402.0,
        # Defaults for analysis (overridable via CLI)
        "keys": ["swact_weighted_average"],
        "steps": ["cmplx", "gener", "synth", "swact"],
    },
    "standard_flowy_transistors_swact": {
        "experiment_name": "multiplier_4bi_8bo_permuti_flowy_bimodal",
        "dir_name_base": "flowy_run_bimodal_12chains_3000steps",
        "origin_output_dir_name": "flowy_run_bimodal_12chains_3000steps_gen_iter0",
        "label": "Flowy Loop (Bimodal) | 12chainsx3000steps | 10kinit-4kgr",
        # Reference lines for swact plots (Bimodal)
        "sme_value": 233.4,
        "tc_value": 402.0,
        "keys": ["swact_weighted_average"],
        "steps": ["cmplx", "gener", "synth", "swact"],
    },
    "default": {
        "experiment_name": "multiplier_4bi_8bo_permuti_allcells_notech_normal_only",
        "dir_name_base": "loop_synth",
        "origin_output_dir_name": "loop_synth_gen_iter0",
        "label": "Yosys Loop (Fullsweep) | Default | 10kinit-4kiter",
        "sme_value": 1e6,
        "tc_value": 1e6,
        # Defaults for analysis (overridable via CLI)
        "keys": ["nb_transistors", "complexity_post_opt"],
        "steps": ["cmplx", "gener", "synth"],
    },
    # Example: add more presets as you like
    # "my_new_run": {
    #     "experiment_name": "...",
    #     "dir_name_base": "...",
    #     "origin_output_dir_name": "...",
    #     "label": "...",
    # },
}


def _as_array_of_axes(axes, n: int):
    """Ensure we always have an indexable array of axes with length n."""
    if n == 1:
        # Matplotlib returns a single Axes for 1 subplot; wrap it.
        return np.array([axes])
    return axes


def main(argv: list[str] | None = None) -> None:
    # logger.disable("genial")  # Disable log calls coming from genial library

    # ----------------------------
    # CLI: choose which run to use
    # ----------------------------
    parser = argparse.ArgumentParser(description="Loop analysis runner")
    parser.add_argument(
        "--run",
        default="default",
        choices=sorted(runs_dict.keys()),
        help="Which preset run configuration to use.",
    )
    parser.add_argument(
        "--n_iter_max",
        type=int,
        default=10,
        help="Max number of iterations to load/aggregate.",
    )
    # (Optional) Expose steps & keys on CLI—kept simple but overridable.
    parser.add_argument(
        "--steps",
        nargs="+",
        default=None,
        help="Pipeline steps to consider (order matters). Overrides run preset if provided.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Metric keys to analyze. Overrides run preset if provided.",
    )
    args = parser.parse_args(argv)

    cfg = runs_dict[args.run]
    experiment_name: str = cfg["experiment_name"]
    dir_name_base: str = cfg["dir_name_base"]
    origin_output_dir_name: str = cfg["origin_output_dir_name"]
    label: str = cfg["label"]

    # Steps/keys: take CLI if provided, otherwise fall back to run preset
    steps: list[str] = args.steps if args.steps is not None else cfg.get("steps", ["cmplx", "gener", "synth"])
    keys_to_analyze: list[str] = args.keys if args.keys is not None else cfg.get("keys", ["nb_transistors"])
    n_iter_max: int = args.n_iter_max

    logger.info(f"Processing run preset: '{args.run}'")
    logger.info(f"  experiment_name={experiment_name}")
    logger.info(f"  dir_name_base={dir_name_base}")
    logger.info(f"  origin_output_dir_name={origin_output_dir_name}")
    logger.info(f"  steps={steps}")
    logger.info(f"  keys_to_analyze={keys_to_analyze}")
    logger.info(f"  n_iter_max={n_iter_max}")

    # Single-call (no list of bases/labels anymore)
    full_set_means, proto_set_means, origin_output_dir_path = get_lists(
        steps=steps,
        n_iter_max=n_iter_max,
        experiment_name=experiment_name,
        dir_name_base=dir_name_base,
        origin_output_dir_name=origin_output_dir_name,
        keys=keys_to_analyze,
    )
    logger.info(f"Origin output dir: {origin_output_dir_path}")

    # Plots for each requested key
    for key in keys_to_analyze:
        do_plots(
            full_set_means,
            proto_set_means,
            key,
            label,
            origin_output_dir_path,
            sme_value=cfg.get("sme_value"),
            tc_value=cfg.get("tc_value"),
        )

    # Save best designs per key
    for key in keys_to_analyze:
        best_dfs: list[pd.DataFrame] = []

        # proto_set_means is (presumably) per-iteration; keep iterating over those
        for proto_set in proto_set_means:
            best_df_i = proto_set.get(f"{key}_best_designs")
            if best_df_i is None:
                continue

            # Plot per-iteration bests if any
            if len(best_df_i) >= 1:
                n_best = len(best_df_i)
                fig, axes = plt.subplots(1, n_best, figsize=(25, 5))
                axes = _as_array_of_axes(axes, n_best)

                for rank, (_, row) in enumerate(best_df_i.iterrows()):
                    plotter.plot_encoding_heatmap_solo(
                        ax=axes[rank],
                        encoding_str=row["encodings_input"],
                        design_number=row["design_number"],
                        bitwidth=proto_set["in_bitwidth"],
                        port_type="input",
                        ax_title=(
                            f"Iter. {row['iteration']} Rank {rank} | dn{row['design_number']}\n{key}  {row[key]:.2E}"
                        ),
                    )

                last_iter = int(best_df_i.iloc[-1]["iteration"])
                fig_path = (
                    Path(origin_output_dir_path)
                    / f"plots_loop_analysis/encoding_viz/iter{last_iter}_best_{key.replace('_', '-')}_encodings_input.png"
                )
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(fig_path)
                plt.close(fig)
                logger.info(f"Saved encoding heatmap(s): {fig_path}")

            best_dfs.append(best_df_i)

        # Concatenate and write parquet for this key
        if best_dfs:
            best_df = pd.concat(best_dfs, axis=0)
            df_path = Path(origin_output_dir_path) / f"plots_loop_analysis/best_designs_{key}.parquet"
            df_path.parent.mkdir(parents=True, exist_ok=True)
            best_df.to_parquet(df_path, index=False)
            logger.info(f"Best designs for key '{key}' saved to {df_path}")
        else:
            logger.warning(f"No best designs found for key '{key}'. Skipping parquet export.")


if __name__ == "__main__":
    main()
