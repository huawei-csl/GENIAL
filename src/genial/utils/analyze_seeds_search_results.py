# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

# %%
from loguru import logger
import argparse
from typing import Any
import json

import pandas as pd
import matplotlib.pyplot as plt
from genial.config.config_dir import ConfigDir
from genial.config.arg_parsers import get_default_parser
import genial.experiment.file_parsers as file_parsers
import genial.utils.utils as utils

plt.rcParams["font.size"] = 15


def parse_args() -> dict[str, Any]:
    # Add arguments for this script
    arg_parser = argparse.ArgumentParser()

    args_dict = vars(arg_parser.parse_known_args()[0])

    # Add standard analyzer arguments
    default_args_dict = get_default_parser()
    args_dict.update(default_args_dict)

    return args_dict


def main():
    args_dict = parse_args()
    dir_config = ConfigDir(is_analysis=True, **args_dict)

    # Get all number of transistors and all seeds
    flattened_data = []
    for synth_design_dirpath in dir_config.synth_out_dir.iterdir():
        seeds_filepath = synth_design_dirpath / "launcher_seeds.json"

        if seeds_filepath.exists():
            design_number = file_parsers.extract_design_number_from_path(seeds_filepath)
            seeds_dict_list = json.load(seeds_filepath.open("r"))
            for d in seeds_dict_list:
                row = {"design_number": design_number, "nb_transistors": d["nb_transistors"], "seeds": d["seeds"]}
                flattened_data.append(row)

    # Build Dataframe
    full_seeds_df = pd.DataFrame(flattened_data)
    best_seeds_df = full_seeds_df.loc[full_seeds_df.groupby("design_number")["nb_transistors"].idxmin()]
    best_seeds_df = best_seeds_df.sort_values("nb_transistors").reset_index()

    # Export the best recipes to a "launcher_seeds.json" file that can be used directly in templates_and_launcher_scripts
    best_seeds_dicts = best_seeds_df[["nb_transistors", "seeds"]].to_dict("records")
    best_seeds_json_filepath = dir_config.analysis_out_dir / "launcher_seeds.json"
    with (best_seeds_json_filepath).open("w") as f:
        f.write(json.dumps(best_seeds_dicts))

    # Log some info
    logger.info(f"{len(best_seeds_df)} design have successfully been searched.")
    logger.info(f"Best designs are the followings:")
    utils.log_dataframe(best_seeds_df[["design_number", "nb_transistors"]])
    logger.info(f"Best seeds combined in the launcher usable file:")
    logger.info(best_seeds_json_filepath)

    # Plot the encodings as tensors
    logger.info(f"Plotting all encodings for human check ...")
    all_encoding_dicts = []
    for dn, nb_trans in zip(best_seeds_df["design_number"], best_seeds_df["nb_transistors"]):
        gener_filepath = file_parsers.get_genered_design_file_path(dir_config.generation_out_dir, dn)
        encodings_dict = file_parsers.extract_encodings(gener_filepath)["input"]

        plot_filepath = gener_filepath.with_name(f"input_encoding_visualisation_dn{dn}.png")
        utils.viz_encoding_as_tensor(
            encodings_dict=encodings_dict,
            fig_title=f"Input Encoding | Design Number {dn} | nb_transistors {nb_trans}",
            filepath=plot_filepath,
            log=False,
        )

        all_encoding_dicts.append(encodings_dict)

    logger.info(f"All encodings plots have been realized and saved in their relative generated designs directories in:")
    logger.info(f"{dir_config.generation_out_dir} /res_<dn>/hdl/input_encoding_visualisation_dn<dn>.png")

    return full_seeds_df


if __name__ == "__main__":
    """
    Script for analysing the results of synth_v4 (seed search)
    Simply use with :
    python src/genial/utils/analyze_seeds_results.py --experiment_name <exp_name> --output_dir_name <output_dir_name>
    """

    df = main()
