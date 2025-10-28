# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

# %%
from pathlib import Path
import shutil
from time import gmtime, strftime
import json

from tqdm import tqdm

from loguru import logger
import pandas as pd

from genial.config.config_dir import ConfigDir
from genial.experiment import file_parsers

from genial.utils.utils import load_database, process_pool_helper, _any_duplicate

import os


def do_purge_designs_main(dir_config: ConfigDir):
    """
    This function find and remove designs with duplicate encodings from the entire output directory.
    Those designs are removed in the files and all (TODO:check this) databases where they might be referenced.
    WARNING: this function actually deletes files, so make sure you know what you are doing before using it.
    """

    encoding_dicts_df, duplicates_df = get_all_encodings_and_duplicates(dir_config)
    to_delete_design_numbers = set(duplicates_df[duplicates_df["has_duplicates"]]["design_number"])

    # Write down list of designs that have been deleted
    with open(dir_config.root_output_dir / "deleted_design_numbers.txt", "a") as f:
        f.write(f"timestamp:{strftime('%Y-%m-%d_%H-%M', gmtime())}\n")
        f.write("\n".join(to_delete_design_numbers))

    logger.info(f"the following file has been updated with list of design numbers to delete:")
    logger.info(dir_config.root_output_dir / "deleted_design_numbers.txt")

    # Actually delete the designs
    precentage = (
        0.0
        if len(to_delete_design_numbers) == 0
        else len(encoding_dicts_df) * 1.0 / len(to_delete_design_numbers) * 100
    )
    logger.warning(
        f"This operation is going to erase {len(to_delete_design_numbers)} designs from the {len(encoding_dicts_df)} existing designs ({precentage:.3f}%)"
    )
    # logger.warning("Do you want to continue?")
    # validation = input("[Y] or [N]")
    # if validation == "Y":
    delete_designs(dir_config=dir_config, design_numbers=to_delete_design_numbers)

    return to_delete_design_numbers


def get_all_encodings_and_duplicates(dir_config: ConfigDir) -> tuple[pd.DataFrame]:
    logger.info(f"Extracting all encodings to find any duplicate ...")

    encoding_dicts_df = file_parsers.read_all_existing_encodings_v2(
        dir_config.root_output_dir, dir_config.bulk_flow_dirname
    )

    logger.info(f"Check all encodings ...")
    with tqdm(
        total=len(encoding_dicts_df), desc=f"x256|Check duplicates and create index dataframe"
    ) as pbar:  # Progress bar
        duplicate_rows = process_pool_helper(
            func=_any_duplicate,
            func_args_gen=((row,) for idx, row in encoding_dicts_df.iterrows()),
            max_workers=dir_config.args_dict.get("nb_workers"),
            pbar=pbar,
        )

    logger.info(f"Assembling duplicate index ...")
    duplicates_df = pd.concat(duplicate_rows, ignore_index=True)
    duplicates_df = duplicates_df.set_index("idx")
    duplicates_df.index.name = None

    return encoding_dicts_df, duplicates_df


def delete_designs(dir_config: ConfigDir, design_numbers: set[str]) -> None:
    for design_number in design_numbers:
        assert isinstance(design_number, str)

        # Delete generated design
        dirpath = dir_config.generation_out_dir / f"res_{design_number}"
        if dirpath.exists():
            shutil.rmtree(dirpath)

        # Delete synthed design
        dirpath = dir_config.synth_out_dir / f"res_{design_number}"
        if dirpath.exists():
            shutil.rmtree(dirpath)

        # Delete swacted design
        dirpath = dir_config.swact_out_dir / f"res_{design_number}"
        if dirpath.exists():
            shutil.rmtree(dirpath)

        logger.info(f"Erased all `res_` directories for design {design_number}")

    # Delete design in synth_df
    filepath = dir_config.analysis_out_dir / "synth_analysis.db.pqt"
    _remove_design_numbers_from_db(filepath, design_numbers)

    # Delete design in swact_df
    filepath = dir_config.analysis_out_dir / "swact_analysis.db.pqt"
    _remove_design_numbers_from_db(filepath, design_numbers)

    # Delete design in encodings_dicts.db
    filepath = dir_config.root_output_dir / "encodings_dicts.db.pqt"
    _remove_design_numbers_from_db(filepath, design_numbers)

    # Delete design in valid_designs.db
    filepath = dir_config.root_output_dir / "valid_designs.db.pqt"
    _remove_design_numbers_from_db(filepath, design_numbers)

    # Delete in dataset_split.json
    filepath = dir_config.trainer_out_root_dir / "dataset_split.json"
    _remove_design_numbers_from_split_dict(filepath, design_numbers)


def _remove_design_numbers_from_db(filepath: Path, design_numbers: set[str]):
    for design_number in design_numbers:
        assert isinstance(design_number, str)

    if filepath.exists():
        df = load_database(filepath)
    else:
        logger.info(f"Database does not exists: {filepath}")
        logger.info(f"It was skipped.")
        return None

    df_filtered = df[~df["design_number"].isin(design_numbers)]
    df_filtered.to_parquet(filepath)
    logger.info(f"Database has been overwritten: {filepath}")

    return None


def _remove_design_numbers_from_split_dict(filepath: Path, design_numbers: set[str]):
    for design_number in design_numbers:
        assert isinstance(design_number, str)

    if not filepath.exists():
        logger.warning(f"json split dictionnary does not exists: {filepath}")
        logger.info(f"It was skipped.")
        return None

    train_split_dict = json.load(open(filepath, "r"))

    new_split_dict = dict()
    for key, design_number_list in train_split_dict.items():
        new_split_dict[key] = list(set(design_number_list) - set(design_numbers))

    json.dump(new_split_dict, fp=open(filepath, "w"), indent=4)
    logger.info(f"Split dictionnary has been overwritten: {filepath}")


# Root dir
experiment_name = "multiplier_4bi_8bo_permuti_allcells_notech_normal_only"
output_dir_name = "loop_synth_gen_iter0"
root_output_dir = Path(os.environ.get["WORK_DIR"]) / "output" / experiment_name / output_dir_name

# Retrieve encoding df
encoding_dicts_df = file_parsers.read_all_existing_encodings_v2(root_output_dir)

# Sort by design number
encoding_dicts_df = encoding_dicts_df.sort_values("design_number").reset_index(drop=True)

# Get duplicates
duplicated_design_nb = encoding_dicts_df[encoding_dicts_df["in_enc_dict"].duplicated()]["design_number"].tolist()

# Create dir variables
generation_out_dir = root_output_dir / "generation_out"
synth_out_dir = root_output_dir / "synth_out"
swact_out_dir = root_output_dir / "swact_out"

# Delete design numbers from dirs
for folder in [generation_out_dir, synth_out_dir, swact_out_dir]:
    for d in duplicated_design_nb:
        dirpath = folder / f"res_{d}"
        print(dirpath)
        if dirpath.exists():
            shutil.rmtree(dirpath)

analysis_out_dir = root_output_dir / "analysis_out"
trainer_out_root_dir = root_output_dir / "trainer_out"

duplicated_design_nb_set = set(duplicated_design_nb)

# Delete design in synth_df
filepath = analysis_out_dir / "synth_analysis.db.pqt"
_remove_design_numbers_from_db(filepath, duplicated_design_nb_set)

# Delete design in swact_df
filepath = analysis_out_dir / "swact_analysis.db.pqt"
_remove_design_numbers_from_db(filepath, duplicated_design_nb_set)

# Delete design in encodings_dicts.db
filepath = root_output_dir / "encodings_dicts.db.pqt"
_remove_design_numbers_from_db(filepath, duplicated_design_nb_set)

# Delete design in valid_designs.db
filepath = root_output_dir / "valid_designs.db.pqt"
_remove_design_numbers_from_db(filepath, duplicated_design_nb_set)

# Delete in dataset_split.json
filepath = trainer_out_root_dir / "dataset_split.json"
_remove_design_numbers_from_split_dict(filepath, duplicated_design_nb_set)
