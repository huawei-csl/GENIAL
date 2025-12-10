from typing import Any

import os
import json
from loguru import logger
import traceback

import shutil

import contextlib
from pathlib import Path

from genial.utils.utils import prepare_temporary_directory, close_temporary_directory
from genial.globals import global_vars
from genial.experiment.file_parsers import get_encoding_dict_from_file

from flowy.flows.reinforce.run.statistical.run_flow import run_flow
from flowy.flows.sim.get_best_design_all_steps import get_best_design
from flowy.data_structures.database import (
    ExperimentIdentifier,
)
from flowy.flows.reinforce.run.statistical.run_flows_in_docker import launch_docker
from flowy.flows.reinforce.run.statistical import run_flow_cmd


@contextlib.contextmanager
def silence_output():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield


def dict_to_args_list(_dict: dict):
    args_list = []
    for key, value in _dict.items():
        if isinstance(value, bool):
            if value:
                args_list.append(f"--{key}")
        else:
            if value is not None:
                args_list.append(f"--{key}")
                args_list.append(str(value))

                if isinstance(value, str):
                    if Path(value).exists():
                        args_list[-1] = "/data" + args_list[-1]

    return args_list


def gen_startup_script_template():
    content = """#!/bin/bash

uv {"--quiet " if $silence_jobs else ""}pip install .

set +e

# OMP_NUM_THREADS for lightgbm
# sleep infinity
PYTHONPATH=$$(pwd) OMP_NUM_THREADS=1 python -u $python_main_path {" ".join($arg_list)} {"> /data/output.txt" if $silence_jobs else "2>&1 | tee /data/output.txt"}
#PYTHONPATH=$$(pwd) OMP_NUM_THREADS=1 python -u $python_main_path {" ".join($arg_list)} > /data/output.txt

# wait forever for debugging
# while true; do sleep 1000; done

mkdir -p /data/$database_folder_parent
${copy_line}

chmod -R 777 /data # to make cleanup possible on docker volume
{"" if $silence_jobs else "ls -altr /data "}

popd > /dev/null
"""

    return content


def setup_launch_args_dict(flowy_run_config: dict, nb_workers: int):
    extra_file_keys = ["verilog_file", "encoding_dict_json_filepath"]
    launch_params_dict = {
        "image_name": "flowy:latest",
        "python_main": run_flow_cmd,
        "up_file_paths": [flowy_run_config[key] for key in extra_file_keys],
        # "silence_jobs": not global_vars["debug"],
        "silence_jobs": True,
        "nb_runs": flowy_run_config["nb_runs"],
        "nb_workers": nb_workers,
        # "commit_hash": "2e4f86da66122edfda67fef4e750c6eb6049835b",
        "startup_script_template": gen_startup_script_template(),
    }

    # Map GENIAL's flowy config yaml file keys to run_flow_cmd args
    key_mappings = {
        "bitwidth_input": "bitwidth",
        "chains": "mockturtle_chains",
        "chain_len": "mockturtle_chain_len",
        "chain_workers": "mockturtle_chain_workers",
    }
    _launch_params_dict = dict()
    for key in key_mappings:
        _launch_params_dict[key_mappings[key]] = flowy_run_config[key]
    for key in flowy_run_config.keys():
        if key not in key_mappings.keys():
            _launch_params_dict[key] = flowy_run_config[key]

    # Clean some keys that are not used by run_flow_cmd
    _launch_params_dict.pop("nb_runs")
    _launch_params_dict.pop("final_selection_metric")
    _launch_params_dict.pop("nb_parallel_runs")

    launch_params_dict["arg_list_job"] = dict_to_args_list(_launch_params_dict)
    print(launch_params_dict)

    return launch_params_dict


def run_flow_wrapper(flowy_run_config: dict, nb_parallel_runs: int):
    try:
        launch_params_dict = setup_launch_args_dict(flowy_run_config, nb_workers=nb_parallel_runs)
        with silence_output():
            result, tmp_dir = launch_docker(
                **launch_params_dict,
                copy_in_src_dir="",
                dont_stream=True,
                # dont_stream=False,
                return_tmp_dir=True,
                debug=global_vars["debug"],
                avoid_copy=True,
                # silence_jobs=True,
            )

    except Exception as e:
        logger.error(f"There was an error running flowy synthesis:")
        logger.error(e)
        logger.error(traceback.format_exc())
        result = None
        tmp_dir = None
        logger.error(f"Skipping copy of the files.")
    return (result, tmp_dir)


class FlowyLauncherHelper:
    _run_flow = run_flow

    def __init__(
        self,
        flowy_run_config: dict[str, Any],
        files_to_copy: list[Path],
        hdl_dirpath: Path,
        output_dir_path: Path,
    ) -> list[Path]:
        self.flowy_run_config = flowy_run_config
        self.tmp_dir = prepare_temporary_directory(files_to_copy)
        self.tmp_dir_path = Path(self.tmp_dir.name)

        # Read generated design's encoding dict and store it in a standalone file
        encoding_dict = get_encoding_dict_from_file(hdl_dirpath / "mydesign_comb.v")

        # Store the encoding as a json file in the temporary directory
        self.encoding_dict_filepath = self.tmp_dir_path / "encoding_dict.json"
        with open(self.encoding_dict_filepath, "w") as f:
            json.dump(encoding_dict, f)

        # Update the flowy_run_config with all the required elements
        for key in self.flowy_run_config:
            if key == "encoding_dict_json_filepath":
                self.flowy_run_config[key] = str(self.encoding_dict_filepath)
            if key == "verilog_file":
                self.flowy_run_config[key] = str(self.tmp_dir_path / "mydesign_hdl" / "mydesign_comb.v")
            if key == "debug" and global_vars["debug"]:
                self.flowy_run_config[key] = True

        # Save some essential output data
        self.design_output_dir_path = output_dir_path
        if not self.design_output_dir_path.exists():
            self.design_output_dir_path.mkdir(parents=True, exist_ok=True)

        return None

    def flowy_synthesis(self):
        # Update environment variables for individual flowy executions
        original_data_dir = os.environ.get("DATA_DIR")
        original_src_dir = os.environ.get("SRC_DIR")
        os.environ["SRC_DIR"] = os.environ.get("SRC_DIR") + "/ext/flowy"
        # os.environ["DATA_DIR"] = str(self.tmp_dir_path)
        os.environ["DATA_DIR"] = str(".")

        nb_parallel_runs = self.flowy_run_config["nb_parallel_runs"]
        nb_workers = "1" if global_vars["debug"] else str(self.flowy_run_config["chain_workers"] * nb_parallel_runs)

        # flowy_tmp_dir = self.tmp_dir_path / "flowy_tmp"

        # Run the flowy flow
        all_run_ids, flowy_tmp_dir = run_flow_wrapper(self.flowy_run_config, nb_parallel_runs=nb_parallel_runs)
        if flowy_tmp_dir is None:
            logger.info(f"The returned `flowy_tmp_dir` was None. I believe it's wrong bro, what about you?")
            return None

        # Find the best design
        experiment_id = ExperimentIdentifier(
            root_database=flowy_tmp_dir.name + "/output/db", experiment=self.flowy_run_config["experiment"]
        )
        os.environ["NB_WORKERS"] = nb_workers

        metric_list = [self.flowy_run_config["final_selection_metric"]]
        if metric_list[0] == "nb_transistors":
            metric_list += ["mockturtle_depth"]
        elif metric_list[0] == "mockturtle_gates":
            metric_list += ["mockturtle_depth"]
        elif metric_list[0] == "swact_count_weighted":
            metric_list += ["mockturtle_depth", "mockturtle_gates"]

        # with silence_output():
        try:
            with silence_output():
                best_run_id, df = get_best_design(
                    experiment_id,
                    target_metric=metric_list,
                    return_record=True,
                    multiproc=not global_vars["debug"],
                )
        except KeyError:
            logger.warning(f"There was an error when analyzing the results of the flowy runs.")
            logger.info(traceback.format_exc())
            logger.info(f"Skipping the copy of the files.")
            return None

        # Save the full data record
        df.to_parquet(self.design_output_dir_path / "flowy_data_record.parquet")

        # Save the data required for GENIAL associated with the best design
        best_data_path = best_run_id.get_path()

        # Get the synthesis-related files
        to_get_paths = [
            "final_mockturtle_design/data_record.json",
            "final_mockturtle_design/mydesign_area_logic.rpt",
            "swact_data/swact_data_final_circuit/data_record.json",
            "tb_data/tb_data_final_circuit/results_db.parquet",
            "tb_files/tb_files_final_circuit/wire_list.json",
            "tb_files/tb_files_final_circuit/mydesign_yosys.v",
            "tb_files/tb_files_final_circuit/mydesign_synth_wrapper.v",
            "tb_files/tb_files_final_circuit/testbench.py",
            "tb_files/tb_files_final_circuit/encoding.json",
            "final_gen_design_files/mydesign_mockturtle_cleaned.v",
        ]
        if not self.design_output_dir_path.exists():
            self.design_output_dir_path.mkdir(parents=True, exist_ok=True)

        for path in to_get_paths:
            _best_data_path: Path = best_data_path / path
            if _best_data_path.exists():
                if _best_data_path.name == "mydesign_mockturtle_cleaned.v":
                    if not (self.design_output_dir_path / "mydesign_yosys.v").exists():
                        shutil.copy(_best_data_path, self.design_output_dir_path / "mydesign_yosys.v")
                    else:
                        shutil.copy(_best_data_path, self.design_output_dir_path)
                elif path == "swact_data/swact_data_final_circuit/data_record.json":
                    shutil.copy(_best_data_path, self.design_output_dir_path / "swact_data_record.json")
                else:
                    shutil.copy(_best_data_path, self.design_output_dir_path)

            else:
                if _best_data_path.parent.exists():
                    for filepath in _best_data_path.parent.iterdir():
                        if _best_data_path.name in filepath.name:
                            shutil.copy(filepath, self.design_output_dir_path)
                else:
                    logger.warning(f"Could not find {path} in {best_data_path}")

        # Restoring environment (in case this was not launched from a subprocess)
        os.environ["SRC_DIR"] = original_src_dir
        if original_data_dir is not None:
            os.environ["DATA_DIR"] = original_data_dir

        # Cleaning Up Temporary Directory
        close_temporary_directory(flowy_tmp_dir)
        close_temporary_directory(self.tmp_dir)

        return self.design_output_dir_path
