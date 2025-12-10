# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Any, Dict
import tempfile
import shutil
import traceback

import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from loguru import logger
import yaml

import concurrent.futures
from numpy import floating
from numpy._typing import _64Bit
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from time import time

import matplotlib.pyplot as plt
import seaborn as sns

import torch

from swact.file_compression_handler import FileCompressionHandler
from genial.globals import global_vars


# Function to log the DataFrame with full log prefix for each line
def log_dataframe(df):
    # Convert the DataFrame to a string
    df_str = df.to_string(index=False)

    # Split into lines
    lines = df_str.split("\n")

    # Log each line with the log prefix
    for line in lines:
        logger.info(line)


def from_binstr_list_to_int_array(list_of_bin_str: list[str] | np.ndarray) -> np.ndarray:
    """
    Convert a list of binary strings to an array of 0s or 1s.
    """
    bin_array = np.fromiter("".join(list_of_bin_str), dtype=np.int64)
    bin_array = bin_array.reshape(len(list_of_bin_str), len(list_of_bin_str[0]))
    return bin_array


def from_int_array_to_binstr_array(array_of_int: np.ndarray, return_as_list: bool = False) -> list[str] | np.ndarray:
    """
    Convert an array of 0s and 1s to a list of binary strings.
    """
    # Make sure the array is filled only with zeros or ones
    assert np.all(np.isin(array_of_int, [0, 1])), "The array must be filled only with zeros or ones"

    array_of_str = np.apply_along_axis(lambda x: "".join(x), 1, array_of_int.astype(str))

    return array_of_str if not return_as_list else array_of_str.tolist()


def negate_list_binstr(list_of_bin_str: list[str]) -> list[str]:
    """
    Invert all `0`s and `1`s in a list of binary strings.
    """
    int_array = from_binstr_list_to_int_array(list_of_bin_str)
    int_array = np.abs(int_array - 1)
    return from_int_array_to_binstr_array(int_array, return_as_list=True)


def enc_dict_to_tensor(val_to_repr_dict: Dict[int, str]) -> torch.Tensor:
    """
    Pipeline to generate the encoding tensors from the encoding (value to representation) dictionary map.
    """
    # Create a list of the dictionary values
    representations = list(val_to_repr_dict.values())
    # Generate representation array
    representation_array = from_binstr_list_to_int_array(representations)
    # Convert to tensor
    return torch.tensor(representation_array, dtype=torch.float32)


def enc_dict_to_values(val_to_repr_dict: Dict[int, str]) -> torch.Tensor:
    """
    Pipeline to get positive only values from the encoding (value to representation) dictionary map.
    """
    # Convert the values to tensors
    values = torch.tensor(list(val_to_repr_dict.keys()), dtype=torch.int32)
    # Rescale for nn.Embedding layer
    return values + torch.abs(torch.min(values))


def viz_encoding_as_tensor(encodings_dict: dict[int:str], fig_title: str, filepath: Path, log: bool = True) -> None:
    """
    Takes an encoding dictionnary {val:repr} and plots it as
    """

    enc_tensor = enc_dict_to_tensor(encodings_dict)
    values = list(encodings_dict.keys())
    bitwidth = enc_tensor.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    sns.heatmap(
        enc_tensor.detach().cpu(),
        ax=ax,
        vmin=0,
        vmax=1,
        xticklabels=False,
        linewidths=0.1,
        linecolor="grey",
        cbar=False,
    )

    xticks = []
    xvals = []
    ax.axvline(0, linewidth=2.5, color="white")
    for i in range(bitwidth):
        xticks.append((i + 0.5))
        xvals.append(i)
        # ax.axvline((i*bitwidth+bitwidth), linewidth=2.5, color="white")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xvals)

    yticks = []
    ax.axvline(0, linewidth=2.5, color="white")
    for i, value in enumerate(values):
        yticks.append((i + 0.5))
        ax.axvline((i * bitwidth + bitwidth), linewidth=2.5, color="white")
    ax.set_yticks(yticks)
    ax.set_yticklabels(values)
    ax.tick_params(axis="x", direction="out", length=5, which="both", colors="black", labelrotation=360)
    ax.tick_params(axis="y", direction="out", length=5, which="both", colors="black", labelrotation=360)

    ax.set_title(fig_title)
    ax.set_xlabel("Bit Position")
    ax.set_ylabel("Values")

    plt.tight_layout()

    plt.savefig(filepath, dpi=300)
    plt.close()
    if log:
        logger.opt(colors=True).info(f"Figure {fig_title} saved in:")
        logger.info(filepath)


def reorder_dataframe(df: pd.DataFrame, column_name: str, order_array: np.ndarray):
    """
    Reorders a pandas DataFrame based on the values in a specific column,
    following the order given in an external array.

    Args:
        df (pd.DataFrame): The DataFrame to reorder.
        column_name (str): The name of the column whose values determine the new order.
        order_array (list-like): The array specifying the desired order of values.

    Returns:
        pd.DataFrame: The reordered DataFrame.
    """

    # Create a categorical dtype from the order array to ensure the specified order
    cat_dtype = pd.CategoricalDtype(categories=order_array, ordered=True)

    # Convert the relevant column to this categorical type
    df[column_name] = df[column_name].astype(cat_dtype)

    # Sort the DataFrame based on the categorical column
    # TODO: what happens to design numbers present df but not in order array?
    return df.sort_values(by=column_name)


def save_serialized_data(npz_filepath: Path, data: Any) -> None:
    # Store serialized data using npz
    serialized_data = pickle.dumps(data)
    np.savez(npz_filepath, data=np.frombuffer(serialized_data, dtype=np.uint8))


def load_serialized_data(npz_filepath: Path) -> Any:
    # Load serialized data using npz and pickle
    # Warning: not secure
    def map_location(storage, loc):
        return storage.cpu()

    torch.serialization.default_restore_location = map_location
    loaded_data = pickle.loads(
        np.load(
            npz_filepath,
        )["data"].tobytes(),
    )
    return loaded_data


def find_maximum_depth(graph: dict[str : list[str]]):
    # TODO: check whether this function is works correctly
    # Initialize a dictionary to store the maximum depth of each node
    max_depth = {}

    # Define a DFS function with memorization
    def dfs(node, visited):
        # If the node is already calculated, return the stored result
        if node in max_depth:
            return max_depth[node]

        # If we detect a cycle (a node is visited again in the current path)
        if node in visited:
            raise ValueError(f"Cycle detected starting at node '{node}'")

        # Mark the node as visited in the current path
        visited.add(node)

        # Initialize the depth for this node as 0
        depth = 0

        # Iterate over the parent nodes
        for parent in graph.get(node, []):
            # Calculate the depth for the parent node recursively
            depth = max(depth, dfs(parent, visited))

        # Remove the node from the current path (backtrack)
        visited.remove(node)

        # Store the calculated depth (+1 for the current node) in the memorization dictionary
        max_depth[node] = depth + 1
        return max_depth[node]

    # Calculate maximum depth for each node in the graph
    for node in graph:
        try:
            dfs(node, set())
        except ValueError as e:
            print(e)
            max_depth[node] = float("inf")

    return max_depth


def prepare_temporary_directory(shared_files: list[str | Path], delete: bool = True) -> tempfile.TemporaryDirectory:
    """
    This function sets up the temporary directory that will be mounted to the docker container as the /app/tmp directory
    """

    tmp_dir = tempfile.TemporaryDirectory(delete=delete)
    tmp_dir_path = Path(tmp_dir.name)

    # Copy all required files into the temporary directory
    for file in shared_files:
        if not file.exists():
            file = FileCompressionHandler.get_compressed_filepath(file)
            if not file.exists():
                raise ValueError(f"File to copy could not be found: {file}")

        if file.is_file():
            copied_filepath = shutil.copy2(file, tmp_dir_path)
            FileCompressionHandler.decompress_if_compressed(Path(copied_filepath))

        elif file.is_dir():
            assert file.name == "hdl"
            copied_dirpath = shutil.copytree(file, tmp_dir_path / "mydesign_hdl")
            for copied_filepath in copied_dirpath.iterdir():
                if copied_filepath.is_file():
                    FileCompressionHandler.decompress_if_compressed(copied_filepath)

    return tmp_dir


def close_temporary_directory(tmp_dir: tempfile.TemporaryDirectory) -> None:
    """Get the useful files from the temporary directory and close it."""

    tmp_dir.cleanup()

    return None


def send_email(subject: str, body: str, calling_function: Any) -> None:
    def is_valid_email(email: str) -> bool:
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return re.match(email_regex, email) is not None

    # Get list of receivers
    receivers_configured = False
    src_dir = os.getenv("SRC_DIR")
    if src_dir is not None:
        receiver_list_yml_path = Path() / ".email_receivers.yml"
        if receiver_list_yml_path.exists():
            receivers_configured = True
    if not receivers_configured:
        logger.warning(f"Tried to send exit job e-mails but list of receivers is not set.")
        logger.info(
            "To set it up, simply configure SRC_DIR in your environement, and add a YAML file `.email_receivers.yml` simply containing a list of e-mail addresses."
        )
    with open(receiver_list_yml_path, "r") as stream:
        receivers = yaml.safe_load(stream)

    # Access credentials from environment variables
    sender_email = os.getenv("GMAIL_ADDRESS")
    app_passwrd = os.getenv("GMAIL_APP_PASSWORD")

    # Assert that receivers are valid e-mail address
    _receivers = []
    for receiver in receivers:
        if not is_valid_email(receiver):
            logger.warning(f"{receiver} is not a valid e-mail address, it will be skipped.")
        else:
            _receivers.append(receiver)

    if len(_receivers) != 0:
        smtpserver = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        smtpserver.ehlo()
        smtpserver.login(sender_email, app_passwrd)

        for receiver_email in receivers:
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = receiver_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            smtpserver.sendmail(sender_email, receiver_email, msg.as_string())

        # Close the connection
        smtpserver.close()

        logger.info(f"Emails sent successfully from {calling_function}.")

    else:
        logger.error(f"All e-mail addresses are wrong, no e-mail were sent.")


def perform_parallel_copy(filepath_lists: list[tuple[Path]], nb_workers: int = 64):
    """
    Perform the copy of directories given as first argument of each tuple of `filepath_lists` (src_dir) to the second argument of the same tuple (tgt_dir)
    Args:
        filepath_lists: each element is a tuple that looks like (src_dir:Path, tgt_dir:Path)
    """

    start_time = time()
    logger.info(f"Copying all generated designs from {filepath_lists[0][0].parent} to {filepath_lists[0][1].parent}")
    final_paths = []
    with tqdm(total=len(filepath_lists), desc=f"x{nb_workers}|Copying Generated Designs") as pbar:  # Progress bar
        with concurrent.futures.ProcessPoolExecutor(max_workers=nb_workers) as executor:
            futures = [executor.submit(shutil.copytree, src_dir, tgt_dir) for src_dir, tgt_dir in filepath_lists]
            for future in concurrent.futures.as_completed(futures):
                final_paths.append(future.result())
                pbar.update(1)

    end_time = time()
    logger.info(
        f"Successfully copied {len(filepath_lists)} directories in {(end_time - start_time) / 60} minutes using {nb_workers} workers."
    )

    return final_paths


def load_database(db_path: Path) -> pd.DataFrame:
    """
    Helper function to open a database file
    Enables stable database reading, with retro-compatibility for csv typed databases
    """

    try:
        df = pd.read_parquet(db_path.with_suffix(".pqt"))
    except Exception as e:
        is_ok = True
        try:
            df = pd.read_csv(db_path.with_suffix(".csv"), dtype=str)
        except UnicodeDecodeError:
            try:
                df = pd.read_parquet(db_path.with_suffix(".csv"))
            except Exception:
                is_ok = False
        except Exception:
            is_ok = False

        if not is_ok:
            logger.error(f"There was an error when loading that file")
            logger.error(f"{db_path}")
            error_msg = traceback.format_exc()
            logger.error(error_msg)
            raise e
        else:
            df.to_parquet(db_path.with_suffix(".pqt"))
            os.remove(db_path.with_suffix(".csv"))
            logger.info(f"CSV database has been replaced by parquet database.")
    return df


def extract_int_string_from_string(string: str) -> str:
    return "".join(filter(str.isdigit, string))


def str_to_int_to_bool(string: str) -> bool:
    return bool(int(string))


def js_distance(series_1: pd.Series, series_2: pd.Series) -> floating[_64Bit]:
    """
    Calculate the JS distance between two series.
    Handles for cases where the two series do not have the same length.
    """
    len_1 = series_1.shape[0]
    len_2 = series_2.shape[0]
    if len_1 == len_2:
        return jensenshannon(series_1, series_2)
    # Resample both distributions to the same number of bins
    num_bins = max(len_1, len_2)
    series_1_hist, bin_edges = np.histogram(series_1, bins=num_bins, density=True)
    series_2_hist, _ = np.histogram(series_2, bins=bin_edges, density=True)
    return jensenshannon(series_1_hist, series_2_hist)


def column_merger_helper(
    source_df: pd.DataFrame, dest_df: pd.DataFrame, cols_to_merge: list[str], ref_col: str = "design_number"
):
    """
    This function maps the values of all the columns to merge from the source dataframe to the destination dataframe
    using ref_col as the reference column.
    """
    cond = None
    for c in cols_to_merge:
        temp_dict = dict(zip(source_df[ref_col], source_df[c]))
        if cond is None:
            cond = dest_df[ref_col].isin(temp_dict)
        dest_df.loc[cond, c] = dest_df.loc[cond, ref_col].map(temp_dict)
    return dest_df


# def process_pool_helper(
#     func,
#     func_args_gen: callable = None,
#     func_kwargs_gen: callable = None,
#     error_message: str = None,
#     max_workers: int = global_vars.get("nb_workers", 16),
#     pbar: tqdm = None,
# ):
#     return_list = []
#     if global_vars.get("debug", False) or max_workers == 1:
#         if func_kwargs_gen is not None:
#             for func_kwargs in func_kwargs_gen:
#                 _res = func(**func_kwargs)
#                 return_list.append(_res)
#         else:
#             for func_args in func_args_gen:
#                 _res = func(*func_args)
#                 return_list.append(_res)
#     else:
#         with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#             # Submit tasks to the executor
#             if func_kwargs_gen is not None:
#                 futures = [executor.submit(func, **func_kwargs) for func_kwargs in func_kwargs_gen]
#             else:
#                 futures = [executor.submit(func, *func_args) for func_args in func_args_gen]
#
#             # Collect results as they become available (order not guaranteed)
#             for future in concurrent.futures.as_completed(futures):
#                 res_row = future.result()
#                 if res_row is not None:
#                     return_list.append(res_row)
#                     if pbar is not None:
#                         pbar.update(1)
#                 elif error_message:
#                     logger.warning(error_message)
#     return return_list

def process_pool_helper(
        func,
        func_args_gen: callable = None,
        func_kwargs_gen: callable = None,
        error_message: str = None,
        max_workers: int = global_vars.get("nb_workers", 16),
        pbar: tqdm = None,
):
    return_list = []
    # if global_vars.get("debug", False) or max_workers == 1:
    if func_kwargs_gen is not None:
        for func_kwargs in func_kwargs_gen:
            _res = func(**func_kwargs)
            return_list.append(_res)
    else:
        for func_args in func_args_gen:
            _res = func(*func_args)
            return_list.append(_res)
    # else:
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #         # Submit tasks to the executor
    #         if func_kwargs_gen is not None:
    #             futures = [executor.submit(func, **func_kwargs) for func_kwargs in func_kwargs_gen]
    #         else:
    #             futures = [executor.submit(func, *func_args) for func_args in func_args_gen]
    #
    #         # Collect results as they become available (order not guaranteed)
    #         for future in concurrent.futures.as_completed(futures):
    #             res_row = future.result()
    #             if res_row is not None:
    #                 return_list.append(res_row)
    #                 if pbar is not None:
    #                     pbar.update(1)
    #             elif error_message:
    #                 logger.warning(error_message)
    return return_list


def combine_dict_list(list_of_dicts):
    """
    Helps combine a list of dictionaries with a single unique key to a single dictionary containing all the keys.
    """
    return {list(d.keys())[0]: list(d.values())[0] for d in list_of_dicts}


def _any_duplicate(row: pd.Series, col_name: str = "in_enc_dict") -> pd.DataFrame:
    """
    This function checks for duplicates in design encodings.
    """
    pattern = r"'(.*?)'"
    encoding_str = row[col_name]
    matches = re.findall(pattern, encoding_str)
    val, counts = np.unique(matches, return_counts=True)
    has_zero = np.any(counts > 1)
    output_row = pd.DataFrame(
        [{"idx": row.name, "design_number": row["design_number"], "has_duplicates": has_zero.item()}]
    )
    return output_row


def add_new_df_to_df(new_df: pd.DataFrame, pre_existing_df: pd.DataFrame, keep_only_preexisting: bool = False):
    """
    This function merges 2 dataframes together by:
     1. Adding columns of the new dataframe to the pre_existing dataframe if they are not already present
     2. Find design_numbers that are not present in the pre_existing dataframe
          In which case those design_numbers are added to the pre_existing dataframe with NaN values for other columns
     3. Update the values of the pre_existing dataframe with the values from the new dataframe at specified design number locations

    """
    if pre_existing_df.empty:
        return new_df

    elif new_df.empty:
        return pre_existing_df

    else:
        # Step 1: Add any new columns from new_synth_df to self.synth_df, filled with NaN
        for col in new_df.columns:
            if col not in pre_existing_df.columns:
                pre_existing_df[col] = pd.NA

        # Step 2: Identify missing design_numbers
        missing_designs = new_df[~new_df["design_number"].isin(pre_existing_df["design_number"])]

        # Step 3: Append missing rows, aligning columns
        if not missing_designs.empty and not keep_only_preexisting:
            shared_columns = list(set(pre_existing_df.columns).intersection(set(missing_designs.columns)))
            pre_existing_df = pd.concat([pre_existing_df, missing_designs[shared_columns]], ignore_index=True)

        # Step 4: Merge/update existing rows
        for c in new_df.columns:
            if c == "design_number":
                continue

            column_map = dict(zip(new_df["design_number"], new_df[c]))
            cond = pre_existing_df["design_number"].isin(column_map)

            pre_existing_df.loc[cond, c] = pre_existing_df.loc[cond, "design_number"].map(column_map)

        return pre_existing_df


def extract_cont_str(string):
    """
    Converts an input string representation of a dictionary into a dictionary.
    Then the string values are concatenated together.
    """
    enc_dict = eval(string)
    out_string = ""
    for v in enc_dict.values():
        out_string += v
    return out_string


def convert_cont_str_to_np(cont_string):
    """
    Converts a string sequences of 0s and 1s to a numpy array.
    """
    return np.array([int(i) for i in cont_string], dtype=np.int16)


def get_twos_complement_dict(n_bits: int) -> dict[int, str]:
    """
    Create a dictionary that maps signed integers to binary strings.
    """
    encoding_dict = {}
    max_val = 2**n_bits
    half = 2 ** (n_bits - 1)

    for i in range(max_val):
        # Interpret as signed two's complement
        signed_val = i - max_val if i >= half else i
        # Format as zero-padded binary string
        bitstring = format(i, f"0{n_bits}b")
        encoding_dict[signed_val] = bitstring

    # Return the sorted dictionary
    return dict(sorted(encoding_dict.items()))


def get_twos_complement_tensor(n_bits: int) -> torch.Tensor:
    return torch.stack(
        [torch.Tensor([int(bit) for bit in bit_string]) for bit_string in get_twos_complement_dict(n_bits).values()]
    )
