# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import matplotlib.pyplot as plt
import numpy as np

# from adder_analysis.experiment.task_analyzer import object
from loguru import logger

from pathlib import Path
import pandas as pd

import seaborn as sns
import ast
from copy import copy

import mplcursors


from tqdm import tqdm
from genial.utils.utils import load_serialized_data

import genial.experiment.file_parsers as file_parsers

from genial.utils.utils import extract_int_string_from_string, process_pool_helper

plt.rcParams["font.size"] = 20

color_map_special = {
    "standard_adder": "red",
    "standard_multiplier": "red",
    "classic_encoding": "chartreuse",
    "default": "blue",
}
color_map_dataset = {
    # "": None,
    # "_weighted": "darkgoldenrod",
    "internal_weighted": "peru",
    "io_weighted": "tan",
}
marker_map_dataset = {
    "": "o",
    "weighted": "o",
    "internal_weighted": "s",
    "io_weighted": "^",
}


def apply_color_map_special(key: str) -> str:
    """
    Wrapper for color map special enabling default color when key is not known in colormap.
    """

    if key not in list(color_map_special.keys()):
        return color_map_special["default"]
    else:
        return color_map_special[key]


def split_specials(analyzer: object, highlight_special: bool) -> tuple[pd.DataFrame]:
    if highlight_special:
        if not analyzer.swact_df.empty:
            swact_df = analyzer.swact_df[~analyzer.swact_df["is_special"]]
            swact_df_spe = analyzer.swact_df[analyzer.swact_df["is_special"]]
        else:
            swact_df = None
            swact_df_spe = None
        synth_df = analyzer.synth_df[~analyzer.synth_df["is_special"]]
        synth_df_spe = analyzer.synth_df[analyzer.synth_df["is_special"]]
    else:
        if not analyzer.swact_df.empty:
            swact_df = analyzer.swact_df
            swact_df_spe = None
        else:
            swact_df = None
            swact_df_spe = None
        synth_df = analyzer.synth_df
        synth_df_spe = None

    return swact_df, swact_df_spe, synth_df, synth_df_spe


def plot_internal_swact_versus_io_swact(
    analyzer: object, plot_type: str = "total", highlight_special: bool = True
) -> None:
    def get_plot_data(
        swact_df: pd.DataFrame, synth_df: pd.DataFrame, test_offset: int, suffix: str, plot_type: str
    ) -> tuple[np.ndarray]:
        if swact_df is None or synth_df is None:
            return None, None, None
        else:
            sub_swact_df = swact_df.iloc[test_offset :: analyzer.test_type_nb]
            if plot_type == "total":
                y_vals = sub_swact_df[f"swact{suffix}_total"].to_numpy()
            elif plot_type == "average":
                y_vals = sub_swact_df[f"per_wire_swact{suffix}_average"].to_numpy()
            elif plot_type == "minimum":
                y_vals = sub_swact_df[f"per_wire_min_swact{suffix}"].to_numpy()
            elif plot_type == "maximum":
                y_vals = sub_swact_df[f"per_wire_max_swact{suffix}"].to_numpy()
            x_vals_wires = sub_swact_df["nb_wires"].to_numpy()
            x_vals_cells = synth_df["nb_cells"].to_numpy()[: len(y_vals)]
            return y_vals, x_vals_wires, x_vals_cells

    logger.info(f"Plotting Internal SwAct vs IO SwAct for '{plot_type}' ...")

    # Dissociate standard designs from special designs
    swact_df, swact_df_spe, synth_df, synth_df_spe = split_specials(analyzer, highlight_special)

    # Loop over different test types
    for i in range(0, analyzer.test_type_nb):
        test_type = swact_df["test_type"].unique()[i]

        ax0: plt.axes
        fig, ax0 = plt.subplots(1, 1, figsize=(15, 15))

        suffix = f"_internal_weighted"
        internal_y_vals, x_vals_wires, x_vals_cells = get_plot_data(swact_df, synth_df, i, suffix, plot_type)
        internal_y_vals_spe, x_vals_wires_spe, x_vals_cells_spe = get_plot_data(
            swact_df_spe, synth_df_spe, i, suffix, plot_type
        )

        suffix = f"_io_weighted"
        io_y_vals, _, _ = get_plot_data(swact_df, synth_df, i, suffix, plot_type)
        io_y_vals_spe, _, _ = get_plot_data(swact_df_spe, synth_df_spe, i, suffix, plot_type)

        scatter = ax0.scatter(internal_y_vals, io_y_vals, c=x_vals_cells)

        if highlight_special:
            for x, y, name in zip(io_y_vals_spe, internal_y_vals_spe, synth_df_spe["special_names"].to_numpy()):
                ax0.scatter(x, y, c=apply_color_map_special(name), label=name, marker=marker_map_dataset["weighted"])

        ax0.legend(fontsize=12)
        per_wire = " per wire" if plot_type in ["average", "minimum", "maximum"] else ""
        ax0.set_xlabel(f"{plot_type.title()} IO (Weighted) SWACT{per_wire}")
        ax0.set_ylabel(f"{plot_type.title()} Internal (Weighted) SWACT{per_wire}")
        ax0.set_title(
            f"{plot_type.title()} Internal (Weighted) SWACT{per_wire} vs IO (Weighted) SWACT{per_wire}\n{test_type} - n={len(internal_y_vals)}\n{analyzer.dir_config.experiment_name}"
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label("Number of Cells")

        l_xlim, r_xlim = ax0.get_xlim()
        b_ylim, t_ylim = ax0.get_ylim()
        hi_lim = max(t_ylim, r_xlim)
        lo_lim = min(b_ylim, l_xlim)
        ax0.set_xlim(left=lo_lim, right=hi_lim)
        ax0.set_ylim(bottom=lo_lim, top=hi_lim)

        plt.tight_layout()
        plt.grid(visible=True, which="both")

        p_wire = "_p_wire" if plot_type in ["average", "minimum", "maximum"] else ""
        fig_path = analyzer.plot_dir / f"{plot_type}_int_vs_io_swact{p_wire}_weighted_{test_type}.png"
        plt.savefig(fig_path, dpi=350)
        logger.opt(colors=True).info(f"<yellow>Plot</yellow> saved at {fig_path}")


def plot_swact_type_versus_area(
    analyzer: object,
    swact_type: str = "total",
    x_axis_type="nb_cells",
    weighted: bool = True,
    marker_size_type: str = "nb_zero_swact",
    colorbar_type: str = "total_worst_swact",
    dataset: str = "",
    highlight_special: bool = True,
    interactive: bool = False,
    scores: np.ndarray = None,
    output_dir_path: Path = None,
    relative: bool = False,
) -> None:
    if analyzer.synth_df.empty or analyzer.swact_df.empty:
        logger.info(f"No data to plot, skipping.")
        return None

    def get_plot_data(
        swact_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        test_offset: int,
        x_axis_type: str,
        suffix: str,
        swact_type: str,
        marker_size_type: str,
        colorbar_type: str,
        interactive: bool = False,
        scores: np.ndarray = None,
    ) -> tuple[np.ndarray]:
        if swact_df is None or synth_df is None or swact_df.empty or synth_df.empty:
            return None, None, None, None, None, None
        elif len(swact_df) == 0:
            return [], [], [], [], None, []
        else:
            if scores is not None and colorbar_type == "scores":
                colorbar_type = "scores"
                logger.info(f"Colorbartype has been overwritten to`scores`.")

            sub_swact_df = swact_df.iloc[test_offset :: analyzer.test_type_nb]
            if swact_type == "total":
                y_vals = sub_swact_df[f"swact{suffix}_total"].to_numpy()
            elif swact_type == "average":
                y_vals = sub_swact_df[f"per_wire_swact{suffix}_average"].to_numpy()
            elif swact_type == "minimum":
                y_vals = sub_swact_df[f"per_wire_min_swact{suffix}"].to_numpy()
            elif swact_type == "maximum":
                y_vals = sub_swact_df[f"per_wire_max_swact{suffix}"].to_numpy()

            # number of wires is not used anymore
            # x_vals_wires = sub_swact_df["nb_wires"].to_numpy()

            ### Get values related to synthesized designs
            # Get max depth
            vals_max_depth = synth_df["max_cell_depth"].to_numpy()[: len(y_vals)]

            # Get number of cells
            x_vals_area = synth_df[x_axis_type].to_numpy()[: len(y_vals)]

            # Get max fanout info
            vals_max_fanout_internal = synth_df["max_fanout_internal"].to_numpy()[: len(y_vals)]
            vals_max_fanout_io = synth_df["max_fanout_io"].to_numpy()[: len(y_vals)]

            ### Prepare values for marker size
            # Get number of patterns that generate zero internal switching activity
            if "fullsweep" in swact_df["test_type"]:
                s_vals_nb_zero_swact = swact_df[swact_df["test_type"] == "fullsweep"][
                    "nb_zero_internal_swact"
                ].to_numpy()
            else:
                s_vals_nb_zero_swact = np.zeros(len(x_vals_area))
            # Replace infinity values (which signifies that the design has not internal wire)
            no_internal_wires_args = np.argwhere(s_vals_nb_zero_swact == float("infinity"))[:, 0]
            if len(no_internal_wires_args) != 0:
                s_vals_nb_zero_swact[no_internal_wires_args] = 3

            if marker_size_type == "nb_zero_swact":
                s_vals = s_vals_nb_zero_swact
            elif marker_size_type == "max_depth":
                s_vals = vals_max_depth
            else:
                valid_marker_size_types = ["nb_zero_swact", "max_depth"]
                raise ValueError(
                    f"`maker_size_type` should be one of: {valid_marker_size_types}, received {marker_size_type}"
                )

            # Rescale marker sizes
            s_vals = (s_vals + 0.5) * 10

            ### Prepare values for colorbar
            # Get worst case internal switching activity
            if "fullsweep" in swact_df["test_type"]:
                c_vals_max_io_swact = swact_df[swact_df["test_type"] == "fullsweep"]["max_io_swact"].to_numpy()
                c_vals_max_io_swact_count = swact_df[swact_df["test_type"] == "fullsweep"][
                    "max_io_swact_count"
                ].to_numpy()
                c_vals_max_internal_swact = swact_df[swact_df["test_type"] == "fullsweep"][
                    "max_internal_swact"
                ].to_numpy()
                c_vals_max_internal_swact_count = swact_df[swact_df["test_type"] == "fullsweep"][
                    "max_internal_swact_count"
                ].to_numpy()

                # Considering that worst case = max SwAct times number of times max SwAct happens
                c_vals_total_swact = (
                    c_vals_max_io_swact * c_vals_max_io_swact_count
                    + c_vals_max_internal_swact * c_vals_max_internal_swact_count
                )
            else:
                c_vals_total_swact = np.zeros(len(x_vals_area))
                c_vals_max_io_swact = np.zeros(len(x_vals_area))
                c_vals_max_internal_swact = np.zeros(len(x_vals_area))

            if colorbar_type == "total_worst_swact":
                c_vals = c_vals_total_swact
            elif colorbar_type == "max_depth":
                c_vals = vals_max_depth
            elif colorbar_type == "scores":
                c_vals = scores
            else:
                valid_colobar_types = ["total_worst_swact", "max_depth"]
                raise ValueError(f"`colorbar_type` should be one of: {valid_colobar_types}, received {colorbar_type}")

            if interactive:
                # Prepare labels for interactive mode
                design_numbers = sub_swact_df["design_number"].to_numpy()
                labels = []
                for idx in range(len(design_numbers)):
                    label = f"dn:{copy(design_numbers[idx])}\ntotal_swact:{copy(y_vals[idx])}\nncells:{copy(x_vals_area[idx])}\nmax_io_swact:{copy(c_vals_max_io_swact[idx])}\nmax_internal_swact:{copy(c_vals_max_internal_swact[idx])}\nnb_zero_swact:{copy(s_vals_nb_zero_swact[idx])}\ntot_worst_swact:{copy(c_vals_total_swact[idx])}\nmax_depth:{copy(vals_max_depth[idx])}\nmax_fanout_io:{copy(vals_max_fanout_io[idx])}\nmax_fanout_internal:{copy(vals_max_fanout_internal[idx])}"
                    labels.append(label)
                labels = np.array(labels)
            else:
                labels = None

            return y_vals, x_vals_area, s_vals, c_vals, labels, no_internal_wires_args

    x_label_title = {
        "nb_cells": "Number of Cells",
        "tot_cell_area": "Total Cell Area",
        "nb_transistors": "Number of Transistors",
    }[x_axis_type]
    logger.info(
        f"Plotting SwAct vs {x_label_title} for '{swact_type}' in 'weighted:{weighted}' and 'relative:{relative}' mode ..."
    )

    # Check dataset
    # if not dataset in ["", "weighted", "internal_weighted", "io_weighted"]:
    # logger.error(f"Unknown required dataset for plot {dataset}")

    # Prepare Suffix and Prefix
    if weighted:
        if dataset == "":
            datasets = ["internal_weighted", "io_weighted", "weighted"]
        else:
            datasets = ["weighted"]
    else:
        datasets = [
            "",
        ]

    # Dissociate standard designs from special designs
    swact_df, swact_df_spe, synth_df, synth_df_spe = split_specials(analyzer, highlight_special)
    if swact_df is None or swact_df.empty:
        logger.warning(f"SwAct DB is empty, returning.")
        return None

    # Loop over different test types
    try:
        for i in range(0, analyzer.test_type_nb):
            test_type = swact_df["test_type"].unique()[i]

            ax0: plt.axes
            fig, ax0 = plt.subplots(1, 1, figsize=(15, 15))

            cluster_handles = []
            highlight_handles = []
            count = 0
            labels = None
            while count < 2:
                for dataset in datasets:
                    # Get the data
                    suffix = f"_{dataset}" if weighted else ""
                    weighted_prefix = f"({dataset.replace('_', ' ').title()})" if weighted else ""
                    if dataset == "weighted":
                        (
                            y_vals,
                            x_vals_area,
                            s_vals_nb_zero_swact,
                            c_vals_total_swact,
                            labels,
                            no_internal_wires_args,
                        ) = get_plot_data(
                            swact_df,
                            synth_df,
                            i,
                            x_axis_type,
                            suffix,
                            swact_type,
                            marker_size_type,
                            colorbar_type,
                            interactive=True,
                        )
                    else:
                        y_vals, x_vals_area, s_vals_nb_zero_swact, c_vals_total_swact, _, no_internal_wires_args = (
                            get_plot_data(
                                swact_df,
                                synth_df,
                                i,
                                x_axis_type,
                                suffix,
                                swact_type,
                                marker_size_type,
                                colorbar_type,
                                interactive=False,
                            )
                        )
                    assert not np.any(np.isnan(y_vals))
                    assert not np.any(np.isnan(s_vals_nb_zero_swact))
                    assert not np.any(np.isnan(c_vals_total_swact))
                    assert not np.any(np.isnan(x_vals_area))
                    y_vals_spe, x_vals_spe_area, s_vals_nb_zero_swact_spe, _, _, no_internal_wires_args_spe = (
                        get_plot_data(
                            swact_df_spe,
                            synth_df_spe,
                            i,
                            x_axis_type,
                            suffix,
                            swact_type,
                            marker_size_type,
                            colorbar_type,
                        )
                    )

                    if relative:
                        if y_vals_spe:
                            overall_minimum = min(min(y_vals), min(y_vals_spe))
                        else:
                            overall_minimum = min(y_vals)

                        if overall_minimum != 0:
                            y_vals = y_vals / overall_minimum * 100
                            try:
                                y_vals_spe = y_vals_spe / overall_minimum * 100
                            except Exception as e:
                                logger.warning(f"Exception while normalizing special designs: {e}")

                    # Plot scatter graph of tot swact versus number of cells
                    if count == 0:
                        if dataset in ["weighted", ""]:
                            if len(no_internal_wires_args) == 0:
                                cluster_handles.append(
                                    ax0.scatter(
                                        x_vals_area,
                                        y_vals,
                                        c=c_vals_total_swact,
                                        label="total",
                                        alpha=0.50,
                                        s=s_vals_nb_zero_swact,
                                    )
                                )
                                main_scatter_handles = cluster_handles[-1]
                            else:
                                cluster_handles.append(
                                    ax0.scatter(
                                        np.delete(x_vals_area, no_internal_wires_args),
                                        np.delete(y_vals, no_internal_wires_args),
                                        c=np.delete(c_vals_total_swact, no_internal_wires_args),
                                        label="total",
                                        alpha=0.50,
                                        s=np.delete(s_vals_nb_zero_swact, no_internal_wires_args),
                                    )
                                )
                                main_scatter_handles = cluster_handles[-1]

                                # Plot no internal wires with specific edgecolors
                                cvmin, cvmax = main_scatter_handles.get_clim()  # Use same colormap
                                cluster_handles.append(
                                    ax0.scatter(
                                        x_vals_area[no_internal_wires_args],
                                        y_vals[no_internal_wires_args],
                                        c=c_vals_total_swact[no_internal_wires_args],
                                        label="total_no_internal_swact",
                                        alpha=0.50,
                                        s=s_vals_nb_zero_swact[no_internal_wires_args],
                                        edgecolors="black",
                                        linestyle="--",
                                        vmin=cvmin,
                                        vmax=cvmax,
                                    )
                                )
                        else:
                            if not interactive:
                                cluster_handles.append(
                                    ax0.scatter(
                                        x_vals_area,
                                        y_vals,
                                        c=color_map_dataset[dataset],
                                        label=dataset,
                                        alpha=0.50,
                                        s=s_vals_nb_zero_swact,
                                    )
                                )

                    # legend_element = ax0.scatter(x_vals_area[0], y_vals[0], c=color_map_dataset[suffix], label=copy(suffix.replace("_", "")))
                    # legend_elements.append(legend_element)
                    if count == 1:
                        try:
                            if highlight_special and not interactive:
                                for x, y, size, name in zip(
                                    x_vals_spe_area,
                                    y_vals_spe,
                                    s_vals_nb_zero_swact_spe,
                                    synth_df_spe["special_names"].to_numpy(),
                                ):
                                    highlight_handles.append(
                                        ax0.scatter(
                                            x,
                                            y,
                                            c=apply_color_map_special(name),
                                            label=name + suffix,
                                            marker=marker_map_dataset[dataset],
                                            s=size,
                                        )
                                    )
                                    # legend_elements.append(legend_element)
                                    # ax0.scatter(x, y, c=apply_color_map_special(name))
                        except Exception as e:
                            logger.warning(f"Exception while highlighting special designs: {e}")
                count += 1

            cluster_handles.extend(highlight_handles)
            ax0.legend(handles=cluster_handles, fontsize=12, loc="upper left")
            ax0.set_xlabel(x_label_title)
            per_wire = " per wire" if swact_type in ["average", "minimum", "maximum"] else ""
            weighted_prefix = "(Weighted) " if weighted else ""
            size_title = {
                "nb_zero_swact": "#0 Int. SwAct",
                "max_depth": "Design Max Depth",
            }[marker_size_type]

            # Setup labels and titles
            y_label = f"{swact_type.title()} {weighted_prefix}SWACT{per_wire}"
            if relative:
                y_label = "Relative " + y_label + " (%)"
            ax0.set_ylabel(y_label)
            title = f"{swact_type.title()} {weighted_prefix}Switching Activity{per_wire} versus {x_label_title}\n{test_type} - n={len(y_vals)} - size:{size_title}\n{analyzer.dir_config.experiment_name}"
            if relative:
                title = "Relative " + title
            ax0.set_title(title)

            cbar = plt.colorbar(main_scatter_handles)
            cbar_title = {
                "total_worst_swact": "Total Worst Cases SwAct",
                "max_depth": "Design Max Depth",
                "scores": "Design Scores (lower is better)",
            }[colorbar_type]
            cbar.set_label(cbar_title)

            plt.tight_layout()
            plt.minorticks_on()
            plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
            plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

            p_wire = "_p_wire" if swact_type in ["average", "minimum", "maximum"] else ""
            filename_weighted_suffix = "_weighted" if weighted else ""
            filename_s = {
                "nb_zero_swact": "nb0swact",
                "max_depth": "mdpth",
            }[marker_size_type]
            filename_c = {"total_worst_swact": "wc_swact", "max_depth": "mdpth", "scores": "scores"}[colorbar_type]

            if output_dir_path is None:
                output_dir_path = analyzer.plot_dir

            subpath = f"it{analyzer.current_iter_nb}_{swact_type}_swact{p_wire}_vs_{x_axis_type}{filename_weighted_suffix}-s_{filename_s}-c_{filename_c}-{test_type}.png"
            if relative:
                subpath = "rel_" + subpath
            fig_path = output_dir_path / subpath
            if not interactive:
                plt.savefig(fig_path, dpi=350)
                logger.opt(colors=True).info(f"<yellow>SWACT vs AREA Plot</yellow> saved at {fig_path}")

            if interactive and swact_type == "total" and weighted:
                # Add Interactive Cursor
                cursor = mplcursors.cursor(main_scatter_handles, hover=False)
                cursor.connect("add", lambda sel: sel.annotation.set_text(labels[sel.index]))
                logger.info(f"Added labels to plot.")
                plt.show()

            # ### Plot zoomed version
            # l_xlim, r_xlim = ax0.get_xlim()
            # new_r_xlim = l_xlim + (r_xlim - l_xlim)/3.0
            # ax0.set_xlim(right=new_r_xlim)
            # b_ylim, t_ylim = ax0.get_ylim()
            # new_t_ylim = b_ylim + (t_ylim - b_ylim)/3.0
            # ax0.set_ylim(top=new_t_ylim)

            # if output_dir_path is None:
            #     output_dir_path = analyzer.plot_dir
            # fig_path = output_dir_path / f"it{analyzer.current_iter_nb}_{swact_type}_swact{p_wire}_vs_{x_axis_type}{filename_weighted_suffix}-s_{filename_s}-c_{filename_c}-{test_type}_zoom.png"
            # if not interactive:
            #     plt.savefig(fig_path, dpi=350)
            #     logger.opt(colors=True).info(f"<yellow>Plot</yellow> saved at {fig_path}")
            plt.close()

            ### Plot point density
            ax0: plt.axes
            fig, ax0 = plt.subplots(1, 1, figsize=(15, 15))
            y, x, _, _, _, _ = get_plot_data(
                analyzer.swact_df,
                analyzer.synth_df,
                i,
                x_axis_type,
                "_weighted",
                swact_type,
                marker_size_type,
                colorbar_type,
                interactive=True,
            )
            density = ax0.hist2d(x, y, bins=(300, 300), cmap="viridis")

            ax0.set_xlabel(x_label_title)
            y_label = f"{swact_type.title()} {weighted_prefix}SWACT{per_wire}"
            if relative:
                y_label = "Relative " + y_label + " (%)"
            ax0.set_ylabel(y_label)
            title = f"{swact_type.title()} [Density] {weighted_prefix}Switching Activity{per_wire} versus {x_label_title}\n{test_type} - n={len(y_vals)} - size:{size_title}\n{analyzer.dir_config.experiment_name}"
            if relative:
                title = "Relative " + title
            ax0.set_title(title)

            cbar = fig.colorbar(density[3], ax=ax0)
            cbar.set_label("Point Density")

            plt.tight_layout()
            plt.minorticks_on()
            plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
            plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

            if output_dir_path is None:
                output_dir_path = analyzer.plot_dir
            subpath = f"it{analyzer.current_iter_nb}_point_density{swact_type}_swact{p_wire}_vs_{x_axis_type}{filename_weighted_suffix}-s_{filename_s}-c_{filename_c}-{test_type}.png"
            if relative:
                subpath = "rel_" + subpath
            fig_path = output_dir_path / subpath
            plt.savefig(fig_path, dpi=350)
            logger.opt(colors=True).info(f"<yellow>DENSITY Plot</yellow> saved at {fig_path}")
    except KeyError as e:
        logger.warning(f"KeyError: {e}")
        return None


def plot_min_max_swact_(analyzer: object, highlight_special: bool = True, weighted: bool = False) -> None:
    if analyzer.synth_df.empty or analyzer.swact_df.empty:
        logger.warning("No data to plot")
        return None

    def get_plot_data(
        swact_df: pd.DataFrame, synth_df: pd.DataFrame, test_offset: int, suffix: str
    ) -> tuple[np.ndarray]:
        if swact_df is None or synth_df is None:
            return None, None, None
        else:
            sub_swact_df = swact_df.iloc[test_offset :: analyzer.test_type_nb]
            min_vals = sub_swact_df[f"per_wire_min_swact{suffix}"].to_numpy()
            max_vals = sub_swact_df[f"per_wire_max_swact{suffix}"].to_numpy()
            return min_vals, max_vals

    # Weighted or not weighted
    suffix = "_weighted" if weighted else ""
    prefix = "(Weighted) " if weighted else ""

    # Dissociate standard designs from special designs
    swact_df, swact_df_spe, synth_df, synth_df_spe = split_specials(analyzer, highlight_special)
    if swact_df is None or swact_df.empty:
        logger.warning(f"SwAct DB is empty, returning.")
        return None

    # Loop over different test types
    try:
        for i in range(0, analyzer.test_type_nb):
            test_type = swact_df["test_type"].unique()[i]

            min_vals, max_vals = get_plot_data(swact_df, synth_df, i, suffix)
            min_vals_spe, max_vals_spe = get_plot_data(swact_df_spe, synth_df_spe, i, suffix)

            # Plot min and max distributions
            combined = np.concatenate([min_vals, max_vals])
            combined_spe = np.concatenate([min_vals_spe, max_vals_spe])
            all_max = np.concatenate([max_vals, max_vals_spe])
            all_min = np.concatenate([min_vals, min_vals_spe])
            total = np.concatenate([combined, combined_spe])
            special_names = synth_df_spe["special_names"].to_numpy()

            fig, ax0 = plt.subplots(1, 1, figsize=(10, 10))
            (n, bins_max, patches) = ax0.hist(all_max, bins=500)
            plt.close()
            fig, ax0 = plt.subplots(1, 1, figsize=(10, 10))
            (n, bins_min, patches) = ax0.hist(all_min, bins=500)
            plt.close()

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
            ax0.hist(min_vals, bins=bins_min, label="min")
            ax1.hist(max_vals, bins=bins_max, label="max")
            for min, max, name in zip(min_vals_spe, max_vals_spe, special_names):
                ax0.hist(min, bins=bins_min, color=apply_color_map_special(name), label=name)
                ax1.hist(max, bins=bins_max, color=apply_color_map_special(name), label=name)

            for idx, axis in enumerate([ax0, ax1]):
                plot_type = ["Minimum", "Maximum"][idx]
                axis.legend()
                # axis.set_xscale('log')
                axis.set_xlabel(f"{prefix}SWACT on the Wire that Switches a {plot_type.title()}")
                axis.tick_params(axis="x", which="both", labelrotation=325)
                axis.set_yscale("log")
                axis.set_ylabel("Nb Occurences")
                axis.set_title(
                    f"{plot_type} {prefix}SWACT on extreme wires\n{test_type} - n={len(total) / 2}\n{analyzer.dir_config.experiment_name}"
                )

            fig_path = analyzer.plot_dir / f"mix_max_swact{suffix}_distrib_{test_type}.png"
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()
            logger.opt(colors=True).info(f"<yellow>Plot</yellow> saved at {fig_path}")

    except KeyError as e:
        logger.warning(f"KeyError: {e}")
        return None


def plot_min_max_fanout_(analyzer: object, highlight_special: bool = True, weighted: bool = False) -> None:
    def get_plot_data(
        swact_df: pd.DataFrame, synth_df: pd.DataFrame, test_offset: int, suffix: str
    ) -> tuple[np.ndarray]:
        if swact_df is None or synth_df is None:
            return None, None, None
        else:
            max_fo_int = synth_df[f"max_fanout_internal"].to_numpy()
            max_fo_io = synth_df[f"max_fanout_io"].to_numpy()
            min_fo_int = synth_df[f"min_fanout_internal"].to_numpy()
            min_fo_io = synth_df[f"min_fanout_io"].to_numpy()
            return max_fo_int, max_fo_io, min_fo_int, min_fo_io

    # Weighted or not weighted
    suffix = "_weighted" if weighted else ""
    prefix = "(Weighted) " if weighted else ""

    # Dissociate standard designs from special designs
    swact_df, swact_df_spe, synth_df, synth_df_spe = split_specials(analyzer, highlight_special)

    # Loop over different test types
    for i in range(0, analyzer.test_type_nb):
        test_type = swact_df["test_type"].unique()[i]

        max_fo_int, max_fo_io, min_fo_int, min_fo_io = get_plot_data(swact_df, synth_df, i, suffix)
        max_fo_int_spe, max_fo_io_spe, min_fo_int_spe, min_fo_io_spe = get_plot_data(
            swact_df_spe, synth_df_spe, i, suffix
        )

        # Plot min and max distributions
        special_names = synth_df_spe["special_names"].to_numpy()

        # fig, ax0 = plt.subplots(1,1, figsize=(10,10))
        # (n, bins_max, patches) = ax0.hist(all_max, bins=500)
        # plt.close()
        # fig, ax0 = plt.subplots(1,1, figsize=(10,10))
        # (n, bins_min, patches) = ax0.hist(all_min, bins=500)
        # plt.close()

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
        (n, bins_min_fo_int, patches) = ax0.hist(min_fo_int, bins=500, label="min_internal")
        (n, bins_min_fo_io, patches) = ax0.hist(min_fo_io, bins=500, label="min_io")
        (n, bins_max_fo_int, patches) = ax1.hist(max_fo_int, bins=500, label="max_internal")
        (n, bins_max_fo_io, patches) = ax1.hist(max_fo_io, bins=500, label="max_io")
        for min_int, max_int, min_io, max_io, name in zip(
            min_fo_int_spe, max_fo_int_spe, min_fo_io_spe, max_fo_io_spe, special_names
        ):
            ax0.hist(min_int, bins=bins_min_fo_int, label=f"min_internal_{name}")
            ax0.hist(min_io, bins=bins_min_fo_io, label=f"min_io_{name}")
            ax1.hist(max_int, bins=bins_max_fo_int, label=f"max_internal_{name}")
            ax1.hist(max_io, bins=bins_max_fo_io, label=f"max_io_{name}")

        for idx, axis in enumerate([ax0, ax1]):
            plot_type = ["Minimum", "Maximum"][idx]
            axis.legend()
            # axis.set_xscale('log')
            axis.set_xlabel(f"{plot_type} {prefix}Fanout of Wire")
            axis.tick_params(axis="x", which="both", labelrotation=325)
            axis.set_yscale("log")
            axis.set_ylabel("Nb Occurences")
            axis.set_title(
                f"{plot_type} {prefix}Fanout on extreme wires\n{test_type} - n={(len(min_fo_int) + len(max_fo_int_spe))}\n{analyzer.dir_config.experiment_name}"
            )

        fig_path = analyzer.plot_dir / f"mix_max_fanout{suffix}_distrib_{test_type}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logger.opt(colors=True).info(f"<yellow>Plot</yellow> saved at {fig_path}")


def plot_min_max_swact_names_(analyzer: object, highlight_special: bool = True) -> None:
    def get_plot_data(swact_df: pd.DataFrame, synth_df: pd.DataFrame, test_offset: int) -> tuple[np.ndarray]:
        if swact_df is None or synth_df is None:
            return None, None, None
        else:
            sub_swact_df = swact_df.iloc[test_offset :: analyzer.test_type_nb]
            min_names = sub_swact_df["per_wire_min_swact_wire_name"].to_numpy()
            max_names = sub_swact_df["per_wire_max_swact_wire_name"].to_numpy()
            return min_names, max_names

    # TODO: function not finished
    # Must be improved

    # Dissociate standard designs from special designs
    swact_df, swact_df_spe, synth_df, synth_df_spe = split_specials(analyzer, highlight_special)

    # Loop over different test types
    for i in range(0, analyzer.test_type_nb):
        test_type = swact_df["test_type"].unique()[i]

        min_vals, max_vals = get_plot_data(swact_df, synth_df, i)
        min_names, min_names_count = np.unique(min_vals, return_counts=True)
        max_names, max_names_count = np.unique(max_vals, return_counts=True)
        # min_vals_spe, max_vals_spe = get_plot_data(swact_df_spe, synth_df_spe, i)

        # Plot min and max distributions
        # combined = np.concatenate([min_vals, max_vals])
        # combined_spe = np.concatenate([min_vals_spe, max_vals_spe])
        # total = np.concatenate([combined, combined_spe])
        # special_names = synth_df_spe["special_names"].to_numpy()

        fig, ax0 = plt.subplots(1, 1, figsize=(20, 20))
        sns.barplot(x=min_names, y=min_names_count, ax=ax0)
        # sns.barplot(x=min_names,y=min_names_count,ax=ax0)
        # (n, bins, patches) = ax0.hist(total, bins=500)
        # plt.close()

        # fig, ax0 = plt.subplots(1,1, figsize=(10,10))
        # ax0.hist(min_vals, bins=bins, color='grey', label="min")
        # ax0.hist(max_vals, bins=bins, color='orange', label="max")
        # for min,max,name in zip(min_vals_spe, max_vals_spe,special_names):
        #     ax0.hist(min, bins=bins, color=apply_color_map_special(name), label=name)
        #     ax0.hist(max, bins=bins, color=apply_color_map_special(name))

        ax0.legend()
        ax0.set_xlabel("Swiching activity on wire")
        # ax0.set_yscale('log')
        ax0.set_ylabel("Nb occurences")
        ax0.set_title(
            f"Max switching activity on extreme wires by names\n{test_type} - n={len(min_names)}\n{analyzer.dir_config.experiment_name}"
        )

        fig_path = analyzer.plot_dir / f"mix_max_swact_distrib_names_{test_type}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logger.opt(colors=True).info(f"<yellow>Plot</yellow> saved at {fig_path}")


def plot_cell_count_distribution(analyzer: object, highlight_special: bool = True, type="nb_transistors") -> None:
    if analyzer.synth_df.empty:
        logger.info(f"Skipping plot_cell_count_distribution because synth_df is empty")
        return None

    swact_df, swact_df_spe, synth_df, synth_df_spe = split_specials(analyzer, highlight_special)

    # Plot distribution of cell counts
    vals = synth_df[type].to_numpy(dtype=np.int64)
    vals_spe = synth_df_spe[type].to_numpy(dtype=np.int64)
    special_names = synth_df_spe["special_names"].to_numpy()
    vals_all = np.concatenate([vals, vals_spe])
    fig, ax0 = plt.subplots(1, 1, figsize=(20, 10))
    (n, bins, patches) = ax0.hist(vals_all, bins=(np.unique(vals_all).shape[0] + 10))
    for val, name in zip(vals_spe, special_names):
        ax0.hist(val, bins=bins, color=apply_color_map_special(name), label=name)

    ax0.legend()
    ax0.set_xlabel(type)
    ax0.set_yscale("log")
    ax0.set_ylabel("Nb occurences")
    ax0.set_title(f"Distribution of {type}\nn={len(vals_all)}\n{analyzer.dir_config.experiment_name}")

    fig_path = analyzer.plot_dir / f"it{analyzer.current_iter_nb}_synth_{type}_distrib.png"

    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
    plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.opt(colors=True).info(
        f"<yellow>Plot</yellow> {type} Count distribution for iteration {analyzer.current_iter_nb} saved at:"
    )
    logger.info(fig_path)


def plot_power_distribution(
    analyzer: object,
) -> None:
    power_df = analyzer.power_df
    if power_df.empty:
        logger.warning(f"Power database is empty, cannot plot any power-related data, returning.")
        return None

    # Plot distribution of cell counts
    vals = power_df["p_comb_dynamic"].to_numpy(dtype=float)
    fig, ax0 = plt.subplots(1, 1, figsize=(15, 10))
    (n, bins, patches) = ax0.hist(vals, bins=(1000))

    ax0.legend()
    ax0.set_xlabel("Power (Watts)")
    ax0.set_yscale("log")
    ax0.set_ylabel("Nb Occurences")
    ax0.set_title(f"Distribution of Power for Designs\nn={len(vals)}\n{analyzer.dir_config.experiment_name}")

    fig_path = analyzer.plot_dir / f"it{analyzer.current_iter_nb}_power_distrib.png"

    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
    plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.opt(colors=True).info(
        f"<yellow>Distribution Plot</yellow> Power for iteration {analyzer.current_iter_nb} saved at:"
    )
    logger.info(fig_path)


def plot_cmplx_distribution(
    analyzer: object,
) -> None:
    cmplx_df = analyzer.cmplx_df
    if cmplx_df.empty:
        logger.warning(f"cmplx database is empty, cannot plot any cmplx-related data, returning.")
        return None

    # Plot distribution of cell counts
    vals = cmplx_df["complexity_post_opt"].to_numpy(dtype=int)
    fig, ax0 = plt.subplots(1, 1, figsize=(15, 10))
    (n, bins, patches) = ax0.hist(vals, bins=(1000))

    ax0.legend()
    ax0.set_xlabel("Complexity (A.U.)")
    ax0.set_yscale("log")
    ax0.set_ylabel("Nb Occurences")
    ax0.set_title(
        f"Distribution of Complexity Post Optimization for Designs\nn={len(vals)}\n{analyzer.dir_config.experiment_name}"
    )

    fig_path = analyzer.plot_dir / f"it{analyzer.current_iter_nb}_cmplx_distrib.png"

    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
    plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.opt(colors=True).info(
        f"<yellow>Distribution Plot</yellow> Complexities for iteration {analyzer.current_iter_nb} saved at:"
    )
    logger.info(fig_path)


def plot_swact_versus_power(analyzer: object) -> None:
    if analyzer.power_df.empty:
        logger.warning(f"power database is empty, cannot plot any cmplx-related data, returning.")
        return None
    if analyzer.synth_df.empty:
        logger.warning(f"synth database is empty, cannot plot any cmplx-related data, returning.")
        return None
    # Plot distribution of cell counts
    power_vals = analyzer.power_df["p_comb_dynamic"]
    swact_vals = analyzer.swact_df["swact_weighted_total"]

    fig, ax0 = plt.subplots(1, 1, figsize=(20, 10))
    scatter = ax0.scatter(swact_vals, power_vals, c=analyzer.synth_df["tot_cell_area"], alpha=0.5)
    ax0.set_xlabel("Total SWACT (Weighted)")
    ax0.set_ylabel("Power (Watts)")
    ax0.set_title(f"SWACT vs Power\n{analyzer.dir_config.experiment_name}\nn={len(swact_vals)}")

    clbr = plt.colorbar(scatter)
    clbr.set_label("Total Cell Area (um2)")

    fig_path = analyzer.plot_dir / f"it{analyzer.current_iter_nb}_swact_versus_power.png"
    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
    plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.opt(colors=True).info(f"<yellow>Plot</yellow> Scatter Power vs Swact saved at:")
    logger.info(fig_path)


def plot_power_versus_area(analyzer: object) -> None:
    from functools import reduce

    if not analyzer.swact_df.empty:
        swact_df = analyzer.swact_df
    else:
        swact_df = None

    if not analyzer.power_df.empty:
        power_df = analyzer.power_df
    else:
        logger.warning(f"Power database is empty, cannot plot any power data, returning.")
        return None

    synth_df = analyzer.synth_df

    all_dfs = []
    for df in [swact_df, power_df, synth_df]:
        if df is not None and not df.empty:
            all_dfs.append(df)

    # Step 1: Find common sorted values in "design_number"
    common_elements = reduce(lambda x, y: x & y, [set(df["design_number"]) for df in all_dfs])
    common_elements = sorted(common_elements)

    # Step 2: Filter and sort each DataFrame by "design_number"
    def align_df(df):
        return (
            df[df["design_number"].astype(str).isin(common_elements)]
            .sort_values(by="design_number")
            .reset_index(drop=True)
        )

    if swact_df is not None:
        swact_df = align_df(swact_df)
    print(power_df)
    power_df = align_df(power_df)
    synth_df = align_df(synth_df)

    # Plot distribution of cell counts
    x_vals = synth_df["tot_cell_area"]
    y_vals = power_df["p_comb_dynamic"]

    fig, ax0 = plt.subplots(1, 1, figsize=(15, 10))
    if not analyzer.swact_df.empty:
        scatter = ax0.scatter(x=x_vals, y=y_vals, c=swact_df["swact_weighted_total"], alpha=0.5)

        clbr = plt.colorbar(scatter)
        clbr.set_label("SwAct")
    else:
        scatter = ax0.scatter(x=x_vals, y=y_vals, c=power_df["max_delay_ps"], alpha=0.5)
        clbr = plt.colorbar(scatter)
        clbr.set_label("Max Delay (ps)")
    ax0.set_ylabel("Power (Watts)")
    ax0.set_xlabel("Total Area (um2)")
    ax0.set_title(f"Power vs Total Cell Area\n{analyzer.dir_config.experiment_name}\nn={len(x_vals)}")

    fig_path = analyzer.plot_dir / f"it{analyzer.current_iter_nb}_power_versus_transistors.png"
    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
    plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.opt(colors=True).info(f"<yellow>Scatter Plot</yellow> Power vs Area at:")
    logger.info(fig_path)


def plot_values_distribution(values: list[int], experiment_name: str, plot_filepath: Path) -> None:
    # Plot distribution of values
    vals = values
    fig, ax0 = plt.subplots(1, 1, figsize=(10, 10))
    (n, bins, patches) = ax0.hist(vals, bins=(np.unique(vals).shape[0] + 10))

    ax0.legend()
    ax0.set_xlabel("Values")
    ax0.set_yscale("log")
    ax0.set_ylabel("Nb occurences")
    ax0.set_title(f"Distribution of values\nn={len(vals)}\n{experiment_name}")

    fig_path = plot_filepath
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.opt(colors=True).info(f"<yellow>Plot</yellow> saved at {fig_path}")


def plot_encoding_heatmaps(analyzer: object, topk_df_dict: dict[str, pd.DataFrame]):
    for test_name, topk_df in topk_df_dict.items():
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        for ax_idx, port_type in enumerate(["input", "output"]):
            ax = axes[ax_idx]

            encodings = topk_df[f"encodings_{port_type}"]
            design_numbers = topk_df[f"design_number"].to_numpy()

            # Convert the list of dictionary into an array
            encoding_arrays = []
            bitwidth = int(analyzer.exp_config[f"{port_type}_bitwidth"])
            for encoding_str in encodings:
                encoding_dict = ast.literal_eval(encoding_str)
                nb_values = len(encoding_dict.values())
                encoding_array = np.zeros((nb_values, bitwidth))
                for val_idx, (key, val) in enumerate(encoding_dict.items()):
                    for char_idx, (char) in enumerate(val):
                        encoding_array[val_idx, char_idx] = int(char)
                encoding_arrays.append(encoding_array)

            try:
                encoding_array_full = np.concatenate(encoding_arrays, axis=1)
            except ValueError:
                logger.warning(
                    f"Tried to concatenate {port_type} encodings arrays of varying sizes for plotting. Please make sure that the encodings are correctly represented in the generated files. (Design numbers:\n{topk_df['design_number']})"
                )
                plt.close()
                return None

            sns.heatmap(
                encoding_array_full,
                ax=ax,
                vmin=0,
                vmax=1,
                xticklabels=False,
                linewidths=0.1,
                linecolor="grey",
                cbar=False,
            )

            ax.set_title(f"{port_type.title()} Representation")

            ax.set_xlabel(f"Design Number")
            ax.set_xticks([(i * bitwidth + bitwidth / 2.0) for i in range(len(design_numbers))])
            ax.set_xticklabels(design_numbers)

            values = list(encoding_dict.keys())
            ax.set_ylabel(f"{port_type.title()} Value")
            yticks = []
            ax.axvline(0, linewidth=2.5, color="white")
            for i, value in enumerate(values):
                yticks.append((i + 0.5))
                ax.axvline((i * bitwidth + bitwidth), linewidth=2.5, color="white")
            ax.set_yticks(yticks)
            ax.set_yticklabels(values)
            ax.tick_params(axis="x", direction="out", length=5, which="both", colors="black", labelrotation=360)
            ax.tick_params(axis="y", direction="out", length=5, which="both", colors="black", labelrotation=360)

        fig.suptitle(
            f"Input and Output Encodings Visualization for TopK Designs\n{test_name} - (MSB=left for fizxed encodings) - bright=1,dark=0"
        )
        plt.tight_layout()

        fig_path = analyzer.plot_dir / f"encoding_heatmap_{port_type}_{test_name}.png"
        plt.savefig(fig_path, dpi=300)
        logger.opt(colors=True).info(f"<yellow>Plot</yellow> saved at {fig_path}")
        plt.close()


def plot_encoding_heatmap_solo(
    ax: plt.Axes,
    encoding_str: dict[str, pd.DataFrame],
    design_number: str,
    bitwidth: int,
    port_type: str = "input",
    ax_title: str = "",
):
    # Convert the list of dictionary into an array
    encoding_dict = ast.literal_eval(encoding_str)
    nb_values = len(encoding_dict.values())
    encoding_array = np.zeros((nb_values, bitwidth))
    for val_idx, (key, val) in enumerate(encoding_dict.items()):
        for char_idx, (char) in enumerate(val):
            encoding_array[val_idx, char_idx] = int(char)

    sns.heatmap(
        encoding_array,
        ax=ax,
        vmin=0,
        vmax=1,
        xticklabels=False,
        linewidths=0.1,
        linecolor="grey",
        cbar=False,
    )

    ax.set_title(ax_title)

    # ax.set_xlabel(f"Design Number")
    # ax.set_xticks([(i * bitwidth + bitwidth / 2.0) for i in range(len(design_numbers))])
    # ax.set_xticklabels(design_numbers)

    values = list(encoding_dict.keys())
    ax.set_ylabel(f"{port_type.title()} Value")
    yticks = []
    ax.axvline(0, linewidth=2.5, color="white")
    for i, value in enumerate(values):
        yticks.append((i + 0.5))
        ax.axvline((i * bitwidth + bitwidth), linewidth=2.5, color="white")
    ax.set_yticks(yticks)
    ax.set_yticklabels(values)
    ax.tick_params(axis="x", direction="out", length=5, which="both", colors="black", labelrotation=360)
    ax.tick_params(axis="y", direction="out", length=5, which="both", colors="black", labelrotation=360)
    plt.tight_layout()

    return ax


def plot_distribution_of_swact(sum_all_swact: np.ndarray, swact_type: str, out_dir_path: Path) -> None:
    """
    Plot the distribution of the sum of the switching activity for all switching patterns
    """
    valid_swact_types = ["io", "internal", "total"]
    assert swact_type in valid_swact_types, f"SwAct type should be among {valid_swact_types}"

    fig, ax0 = plt.subplots(1, 1, figsize=(10, 10))
    ax0.hist(sum_all_swact, bins=500)

    design_number = extract_int_string_from_string(out_dir_path.name)
    ax0.set_title(f"Distribution of {swact_type.title()} SWACT\ndesign number:{design_number}.")
    ax0.set_xlabel(f"{swact_type} SWACT per input transition")
    ax0.set_ylabel(f"Nb Occurences")

    plt.tight_layout()
    plt.grid(visible=True, which="both")

    fig_path = out_dir_path / f"full_swact_{swact_type}_dn{design_number}.png"
    plt.savefig(fig_path, dpi=300)
    # logger.opt(colors=True).info(f"<yellow>Plot</yellow> saved at {fig_path}")
    plt.close()


def plot_internal_versus_io_swact_count(
    all_swact_overview_df: pd.DataFrame, design_number: int, experiment_name: int, out_dir_path: Path
) -> None:
    # logger.info(f"Plotting Internal SwAct vs IO SwAct for design {design_number} ...")

    io_swact_vals = all_swact_overview_df["io_swact_count"].to_numpy(dtype=int)
    internal_swact_vals = all_swact_overview_df["internal_swact_count"].to_numpy(dtype=int)

    ax0: plt.axes
    fig, ax0 = plt.subplots(1, 1, figsize=(15, 15))

    scatter = ax0.scatter(x=io_swact_vals, y=internal_swact_vals, c=(io_swact_vals + internal_swact_vals))

    ax0.set_xlabel(f"IO SWACT Counts")
    ax0.set_ylabel(f"Internal SWACT Count")
    ax0.set_title(f"Internal SWACT vs IO SWACT\ndesign number:{design_number}\n{experiment_name}")

    clbr = plt.colorbar(scatter)
    clbr.set_label("Sum: Internal + IO SwAct")

    plt.tight_layout()
    plt.grid(visible=True, which="both")

    fig_path = out_dir_path / f"int_vs_io_swact_count_dn{design_number}.png"
    plt.savefig(fig_path, dpi=300)
    # logger.opt(colors=True).info(f"<yellow>Plot</yellow> saved at {fig_path}")
    plt.close()


def _read_histogram_data_from_npz(dir_path: Path, test_type_name: str, valid_design_numbers: list[str]) -> dict:
    design_number = extract_int_string_from_string(dir_path.name)

    # Open histogram dictionariy of current test type
    npz_filepath = dir_path / f"test_val_hist_{test_type_name}.npz"
    if not npz_filepath.exists():
        # logger.error(f"File {npz_filepath} does not exist")
        return None
    hist_data = load_serialized_data(npz_filepath)

    results_rows_list = []
    for operand in hist_data.keys():
        res_dict = {"design_number": design_number, "operand": operand}
        if design_number in valid_design_numbers:
            try:
                vals = np.array(hist_data[operand]["vals"], dtype=np.int64)
                count = np.array(hist_data[operand]["count"], dtype=np.int64)

                sorted_args = np.argsort(vals)
                sorted_count = count[sorted_args]
                sorted_vals = vals[sorted_args]

                res_dict.update({val: count for val, count in zip(sorted_vals, sorted_count)})
                results_rows_list.append(res_dict)

            except Exception as e:
                logger.error(design_number)
                logger.error(e)
                return None

    return pd.DataFrame(results_rows_list)


def plot_test_type_value_histogram(analyzer: object, test_type_names: list[str]) -> None:
    logger.info(f"Plotting total histogram of values for tests {test_type_names} ...")
    valid_design_numbers = file_parsers.get_list_of_swact_designs_number(analyzer.dir_config)

    if not (analyzer.dir_config.analysis_out_dir / "swact_analysis").exists():
        logger.warning("Directory for swact analysis does not exist. Skipping histogram plot.")
        return

    # Open all histogram dictionaries
    for test_type_name in test_type_names:
        all_swact_dir = (analyzer.dir_config.analysis_out_dir / "swact_analysis").iterdir()

        if analyzer.is_debug:
            all_swact_dir = list(all_swact_dir)[:3]

        pbar_desc = (
            f"x{analyzer.nb_workers}|{analyzer.experiment_name} | "
            f"Get total histogram of values for test {test_type_name}"
        )
        nb_designs = 2 if analyzer.is_debug else len(valid_design_numbers)
        with tqdm(
            total=nb_designs,
            desc=pbar_desc,
        ) as pbar:  # Progress bar
            res_list = process_pool_helper(
                func=_read_histogram_data_from_npz,
                func_args_gen=((dir_path, test_type_name, valid_design_numbers) for dir_path in all_swact_dir),
                pbar=pbar,
            )

        res_list = [res for res in res_list if res is not None]
        if len(res_list) == 0:
            logger.warning(f"No histogram data file found for test {test_type_name}")
            logger.info(f"Skipping.")
            return None

        logger.info(f"Consolidating results for test {test_type_name} ...")
        combined_df = pd.concat(res_list, ignore_index=True)

        # Draw Plots
        axes: list[plt.axes]
        nb_operands = len(combined_df["operand"].unique())
        fig, axes = plt.subplots(1, nb_operands, figsize=(nb_operands * 10, 10))

        for idx, operand in enumerate(combined_df["operand"].unique()):
            sub_df = combined_df[combined_df["operand"] == operand].drop(["design_number", "operand"], axis=1)
            # TODO: drop columns filled with NaN (where everything is NaN it means that value is not associated with this operand)

            counts = sub_df.mean().to_numpy()
            values = sub_df.columns.to_numpy()

            axes[idx].bar(values, counts, color="skyblue", edgecolor="black")
            axes[idx].set_title(f"Operand {operand}")
            axes[idx].set_xlabel(f"Value")
            axes[idx].set_ylabel(f"Nb Occurences")
            axes[idx].grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)

        fig.suptitle(
            f"IO Value Distribution (Averaged over all designs)\n{test_type_name} - n={len(sub_df)}\n{analyzer.dir_config.experiment_name}"
        )
        plt.tight_layout()

        fig_path = analyzer.plot_dir / f"test_vals_hist_{test_type_name}.png"
        plt.savefig(fig_path, dpi=300)
        logger.opt(colors=True).info(f"<yellow>Plot</yellow> saved at {fig_path}")
        plt.close()


def plot_swact_cost_vs_encoder_design_coefficient(
    combined_switch_cost_df: pd.DataFrame,
    curr_design_swact_outdir: Path,
    design_number: str,
    encoder_number: str,
    experiment_name: str,
    test_type_name: str,
    is_weighted: bool,
):
    x_vals = []

    y_dne_vals_io = []
    y_dne_vals_internal = []
    y_dne_vals_total = []

    y_donly_vals_io = []
    y_donly_vals_internal = []
    y_donly_vals_total = []

    for reuse_coefficient in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        x_vals.append(reuse_coefficient)

        y_dne_vals_io.append(combined_switch_cost_df[f"dne_r{reuse_coefficient}_io_swact_cost"].astype(float).sum())
        y_dne_vals_internal.append(
            combined_switch_cost_df[f"dne_r{reuse_coefficient}_internal_swact_cost"].astype(float).sum()
        )
        y_dne_vals_total.append(
            combined_switch_cost_df[f"dne_r{reuse_coefficient}_total_swact_cost"].astype(float).sum()
        )

        y_donly_vals_io.append(combined_switch_cost_df[f"design_io_swact_cost"].astype(float).sum())
        y_donly_vals_internal.append(combined_switch_cost_df[f"design_internal_swact_cost"].astype(float).sum())
        y_donly_vals_total.append(combined_switch_cost_df[f"design_total_swact_cost"].astype(float).sum())

    graphs = [
        (x_vals, y_dne_vals_io, "D+E IO SwAct Cost", "o", "teal", True),
        (x_vals, y_dne_vals_internal, "D+E Internal SwAct Cost", "o", "darkgoldenrod", True),
        (x_vals, y_dne_vals_total, "D+E Total SwAct Cost", "o", "darkmagenta", True),
        (x_vals, y_donly_vals_io, "D IO SwAct Cost", "x", "lightseagreen", False),
        (x_vals, y_donly_vals_internal, "D Internal SwAct Cost", "x", "goldenrod", False),
        (x_vals, y_donly_vals_total, "D Total SwAct Cost", "x", "darkviolet", False),
    ]

    ax0: plt.axes
    fig, ax0 = plt.subplots(1, 1, figsize=(15, 15))

    for graph in graphs:
        ax0.plot(graph[0], graph[1], label=graph[2], marker=graph[3], color=graph[4])

        for x, y in zip(graph[0], graph[1]):
            ax0.text(x, y, f"  {y:.2E}", fontsize=9, rotation=45)
            if not graph[5]:
                # Annotate a single point
                break

    weighted_title = "Weighted " if is_weighted else ""
    ax0.set_ylabel(f"{weighted_title}Switch Activity Cost")
    ax0.set_xlabel(f"Encoder Reuse Ratio")
    ax0.set_title(
        f"{weighted_title}SwAct Cost vs Encoder Reuse\n{experiment_name}\n{test_type_name} - design {design_number} - encoder {encoder_number}"
    )

    ax0.set_xscale("log")
    ax0.minorticks_on()

    plt.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
    plt.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

    plt.tight_layout()
    filepath = curr_design_swact_outdir / f"design_n_encoder_swat_cost_{test_type_name}_d{design_number}.png"
    plt.savefig(filepath)
    plt.close()


def plot_max_fanout_versus_max_depth(analyzer: object) -> None:
    logger.info(f"Plotting Max Fanout vs Max Depth for all designs ...")

    if analyzer.synth_df.empty:
        logger.warning("Synthesise DB is empty. Skipping plot ...")
        return

    max_fanout_internal = analyzer.synth_df["max_fanout_internal"].to_numpy(dtype=int)
    max_fanout_io = analyzer.synth_df["max_fanout_io"].to_numpy(dtype=int)
    max_depth = analyzer.synth_df["max_cell_depth"].to_numpy(dtype=int)

    ax0: plt.axes
    ax1: plt.axes
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 7.5))

    ax0.scatter(x=max_depth, y=max_fanout_internal, label="max internal fanout")
    ax1.scatter(x=max_depth, y=max_fanout_io, label="max io fanout")

    ax0.set_ylabel(f"Max Internal Fanout")
    ax0.set_xlabel(f"Max Depth")
    ax0.set_title(f"Max Internal Fanout")

    ax1.set_ylabel(f"Max IO Fanout")
    ax1.set_xlabel(f"Max Depth")
    ax1.set_title(f"Max IO Fanout")

    for axis in [ax0, ax1]:
        axis.grid(visible=True, which="major", color="grey", linestyle="-", alpha=0.5)
        axis.grid(visible=True, which="minor", color="grey", linestyle="--", alpha=0.2)

    fig.suptitle(f"Max Fanout vs Max Cell Depth\n{analyzer.experiment_name}\nn={len(max_depth)}")
    plt.tight_layout()

    # clbr = plt.colorbar(scatter)
    # clbr.set_label("Sum: Internal + IO SwAct")

    fig_path = analyzer.plot_dir / f"max_fanout_vs_max_depth.png"
    plt.savefig(fig_path, dpi=300)
    logger.opt(colors=True).info(f"<yellow>Plot</yellow> saved at {fig_path}")
    plt.close()
