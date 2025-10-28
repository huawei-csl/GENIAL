# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from typing import Any, TYPE_CHECKING
from loguru import logger

import tempfile
from string import Template

import os
import shutil
from pathlib import Path
from argparse import ArgumentParser
import traceback
import yaml

from time import time
from datetime import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

# Docker is optional at import time. Import types only for type checkers,
# and import the runtime module lazily where needed.
if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from docker.models.containers import Container  # type: ignore
    from docker.models.volumes import Volume  # type: ignore
else:  # Fallbacks so annotations still work at runtime without docker installed
    Container = Any  # type: ignore
    Volume = Any  # type: ignore

from genial.experiment.task_generator import DesignGenerator
from genial.config.config_dir import ConfigDir
from genial.config.arg_parsers import get_default_parser

from genial.experiment import file_parsers
from genial.utils import utils
from genial.utils.sample_output_dir import (
    split_design_numbers_in_bins_on_tgt_metric,
    extend_list_of_selected_designs,
)
from genial.experiment.loop_module import LoopModule
from genial.globals import global_vars

from swact.file_compression_handler import FileCompressionHandler
from swact.netlist import format_synthed_design, get_wire_list

if global_vars["flowy_available"]:
    from genial.ext_plugins.flowy.flowy_launcher_helper import FlowyLauncherHelper


class Launcher(LoopModule):
    def __init__(
        self,
        dir_config: ConfigDir | None = None,
        docker_image_name: str | Path = f"genial:latest",
        **kwargs,
    ) -> None:
        """
        This class is used to launch the experiments in a docker container
        Note: kwargs has precedence over all other configuration methods
        """
        super().__init__()

        logger.info(f"Setting up Launcher ...")

        # Initialize the experiment configuration
        if dir_config is None:
            self.args_dict = Launcher.parse_args()
            self.dir_config = ConfigDir(is_analysis=False, **self.args_dict)
        else:
            self.args_dict = dir_config.args_dict
            self.dir_config = dir_config

        self.args_dict.update(**kwargs)

        # Default configuration
        self.docker_image_name = docker_image_name

        # Memorize tasks to be completed
        self.skip_synth = self.args_dict.get("skip_synth", False)
        self.skip_swact = self.args_dict.get("skip_swact", False) or self.args_dict.get("synth_only", False)
        self.skip_power = self.args_dict.get("skip_power", False) or self.args_dict.get("synth_only", False)
        self.init_log_files()
        # self.args_dict.get("skip_already_synth", False) = self.args_dict.get("skip_already_synth", False)
        # self.args_dict.get("skip_already_swact", False) = self.args_dict.get("skip_already_swact", False)

        # Remove switching activity measure of wires not used in the design?
        self.remove_dangling_wires = False  # No, because unused activity is still activity

        # Break iterations after a single iteration
        is_debug = self.args_dict.get("debug", False)
        if is_debug is not None and is_debug:
            self.is_debug = is_debug
            self.max_files_debug = 2
            self.nb_workers = 1
            # logger.debug("DEBUG mode is ON. A single file with be synthesized and tested")
        else:
            self.is_debug = False
            self.max_files_debug = None
            nb_workers = self.args_dict.get("nb_workers", None)
            self.nb_workers = 64 if nb_workers is None else nb_workers
            logger.opt(colors=True).info(
                f"Tasks will be realized in parallel with <red>{self.nb_workers} workers</red>"
            )

        self.user_design_number_list = self.args_dict.get("design_number_list", None)
        if self.user_design_number_list is not None and len(self.user_design_number_list) > 0:
            logger.info(
                f"List of design numbers has been passed as argument. Only those will be treated. Full list: {self.user_design_number_list}"
            )

        # Current Experiment configuration
        self.exp_config = self.dir_config.get_experiment_configuration()
        self.synth_instance_names = self.exp_config["top_synth_instance_names"]
        self.experiment_name = self.args_dict.get("experiment_name")

        # Prepare flowy arguments in case

        # Prepare other synth dir_config if "other_output_dirpath" has been provided as argument
        self.other_synth_dir_config = ConfigDir.setup_other_dir_config(self.args_dict)

        logger.info(f"Launcher initialized.\n\n")

    def init_log_files(self):
        """This function restarts all log files"""

        if self.skip_synth:
            logfile_name = "launcher_swact.log"
        elif self.skip_swact:
            logfile_name = "launcher_synth.log"
        else:
            logfile_name = "launcher_full.log"

        self.logdir = self.dir_config.root_output_dir / "logs"
        logger.info(f"Log directory prepared in {self.logdir}")

        if not self.logdir.exists():
            self.logdir.mkdir()

        log = self.logdir / logfile_name
        with log.open("a") as f:
            f.write("\n\n\n")

        logger.add(log, level="INFO")

    @staticmethod
    def get_result_files(tmp_dir: tempfile.TemporaryDirectory, design_number: str, task_output_dir_path: Path) -> Path:
        """
        This function creates the result directory for the design_number and copy all results files from the tmp_dir (the directory where docker tasks store their results)
        """

        res_dir_path = task_output_dir_path / f"res_{design_number}"
        tmp_results_path = Path(tmp_dir.name) / "synth_results"

        if not res_dir_path.exists():
            res_dir_path.mkdir(parents=True)

        # Copy all files from tmp_dir_path / syn
        for file in tmp_results_path.iterdir():
            file_path = file
            if file_path.is_file():
                copied_file = shutil.copy(file_path, res_dir_path / file.name)
                if file_path.suffix == ".v":
                    file_path = FileCompressionHandler.compress_file(copied_file)

        return res_dir_path

    @staticmethod
    def run_script_in_container(
        design_number: str,
        files_to_copy: list[str | Path],
        launch_script_name: str,
        task_output_dir_path: Path,
        docker_image_name: str,
        docker_container_name: str,
    ) -> Path:
        """Runs the given script content within a Docker container.

        Args:
            kwargs: Additional arguments to pass to container.run()

        Returns:
            The path to the output directory.
        """
        # with semaphore:  # Acquire the semaphore before running the task
        # try:

        # Setup temporary Directory
        tmp_dir = utils.prepare_temporary_directory(files_to_copy)

        # Mount the temporary directory with the docker container
        container = None
        volume = None
        try:
            container, volume = Launcher._docker_wrapper(
                image_name=docker_image_name,
                container_name=docker_container_name,
                tmp_dir_obj=tmp_dir,
                launch_script_name=launch_script_name,
            )
        except Exception as e:
            logger.error(
                f"Design number {design_number} failed to run in docker container. Skipped. Copied files were:"
            )
            for file in files_to_copy:
                logger.error(file)
            logger.error(str(e))
            res_dir_path = None
            Launcher.get_result_files(
                tmp_dir=tmp_dir, design_number=design_number, task_output_dir_path=task_output_dir_path
            )
        else:
            # Get all result files
            res_dir_path = Launcher.get_result_files(
                tmp_dir=tmp_dir, design_number=design_number, task_output_dir_path=task_output_dir_path
            )

        # Close temporary directory and get useful data
        Launcher.cleanup(container, volume)
        utils.close_temporary_directory(tmp_dir)

        # finally:
        # semaphore.release()  # Release the semaphore when done

        return res_dir_path

    @staticmethod
    def cleanup(container: Container | None, volume: Volume | None):
        try:
            container.stop()
        except Exception:
            pass
        try:
            container.remove()
        except Exception:
            pass
        try:
            volume.remove(force=True)
        except Exception as e:
            logger.error(e)
            pass

    @staticmethod
    def _docker_wrapper(
        image_name: str, container_name: str, tmp_dir_obj: tempfile.TemporaryDirectory, launch_script_name: str
    ) -> Any:
        """
        Wrapper for the docker run command.
        """
        # Make safety checks
        assert launch_script_name is not None, "launch_script_name must be provided to the docker_wrapper."

        # Setup a docker client
        # import os
        # os.environ['DOCKER_HOST'] = f'unix:///{os.environ.get("XDG_RUNTIME_DIR")}/docker.sock'
        # Import docker lazily so the module can be imported without docker installed/running
        import docker  # type: ignore

        client = docker.from_env()

        # Create a volume that attaches to the temporary directory
        volume_name = container_name + "_volume"
        volume = client.volumes.create(
            name=volume_name,
            driver="local",
            driver_opts={
                "type": "none",
                "o": "bind",
                "device": str(Path(tmp_dir_obj.name)),
            },
        )

        # Mount the temporary file volume
        volumes = {volume_name: {"bind": "/app/tmp", "mode": "rw"}}
        command = ["bash", f"/app/tmp/{launch_script_name}.sh"]
        # logger.info(command)

        # if launch_script_name == "launch_script_evaluation_switching_activity":
        #     command = ["sleep", "infinity"]
        # if launch_script_name == "launch_script_power_extraction":
        #     command = ["sleep", "infinity"]

        # Run the container with command to execute the program
        container = client.containers.run(
            image=image_name,
            volumes=volumes,
            # user=os.getuid(),
            # group_add=[
            #     os.getgid(),
            # ],
            name=container_name,
            working_dir="/app",
            command=command,
            # command=["sleep", "infinity"],
            remove=True,
            # cpu_period=100000,  # 100 ms per period # TODO: make this parameterizable for SLURM Cluster execution
            # cpu_quota=100000,   # 100% of 1 CPU
        )

        return container, volume

    # from queue import Queue

    # # Initialize queue (stays the same)
    # res_dir_queue = Queue()

    # # Setup data shared among jobs (stays the same)
    # shared = launcher.get_shared_data()
    @staticmethod
    def process_batch(
        file_batch: tuple[Path],
        shared_files: list[Path],
        launch_script_name: str,
        task_output_dir_path: Path,
        docker_image_name: str,
        top_pid: str,
        *args,
        **kwargs,
    ) -> Path:
        # Extract design_number out of file_batch
        design_number = file_parsers.extract_design_number_from_path(file_batch[0])
        all_files = shared_files + list(file_batch)
        docker_container_name = f"{os.environ.get('USER')}_exp_{design_number}_{launch_script_name}_{top_pid}"
        try:
            results = Launcher.run_script_in_container(
                design_number,
                all_files,
                launch_script_name,
                task_output_dir_path,
                docker_image_name,
                docker_container_name,
            )
        except Exception:
            results = None
            logger.error(f"There was an error with design number: {design_number} when launching {launch_script_name}")
            logger.error(traceback.format_exc())

        return results

    @staticmethod
    def flowy_process_batch(
        file_batch: tuple[Path],
        shared_files: list[Path],
        launch_script_name: str,
        task_output_dir_path: Path,
        docker_image_name: str,
        top_pid: str,
        flowy_run_config: dict[str, Any],
    ) -> Path:
        # Extract design_number out of file_batch
        design_number = file_parsers.extract_design_number_from_path(file_batch[0])
        dn_output_dirapth = task_output_dir_path / f"res_{design_number}"
        if not dn_output_dirapth.exists():
            dn_output_dirapth.mkdir(parents=True, exist_ok=True)
        all_files = shared_files + list(file_batch)
        retries = 3

        # docker_container_name = f"{os.environ.get('USER')}_exp_{design_number}_{launch_script_name}_{top_pid}"
        # while retries > 0:
        for retry_count in range(retries):
            try:
                flowy_launcher_helper = FlowyLauncherHelper(
                    flowy_run_config=flowy_run_config,
                    files_to_copy=all_files,
                    hdl_dirpath=file_batch[0],
                    output_dir_path=dn_output_dirapth,
                )
                results = flowy_launcher_helper.flowy_synthesis()
                if results is None:
                    logger.error(
                        f"There was an error with design number: {design_number} when launching {launch_script_name}"
                    )
                    # retries -= 1
                else:
                    # retries = 0
                    break

            except Exception as e:
                results = None
                # retries -= 1
                # if retries == 0:
                if retry_count == retries - 1:
                    logger.error(
                        f"There was an error with design number: {design_number} when launching {launch_script_name}"
                    )
                    logger.error(e)
                    logger.error(traceback.format_exc())

        return results

    __valid_launch_tasks__ = ["synthesis", "synthesis_flowy", "evaluation_switching_activity", "power_extraction"]

    def __launch_all(
        self,
        shared_files: list[str | Path],
        batched_files: list[list[str | Path]],
        task: str,
    ) -> list[Path]:
        """
        Launches all files in the batched_files list to the docker container.

        ## Args:
            - shared_files: list of files that are shared between all designs
            - batched_files: list of lists of files that are batched together
            - task: must be in {Launcher.__valid_launch_tasks__}
        """

        assert task in Launcher.__valid_launch_tasks__, f"Task specified not in available tasks set."

        # Initialize queue
        end = len(batched_files)
        if global_vars.get("debug"):
            end = self.max_files_debug
            logger.warning(f"~Task {task} has been limited to {end} designs because the debug flag is set to")

        # Setup Defaults
        task_output_dir_path = None
        docker_image_name = self.docker_image_name
        launcher_func = Launcher.process_batch
        flowy_run_config_dict = None
        first_level_parallel = self.nb_workers

        if task == "synthesis":
            task_output_dir_path = self.dir_config.synth_out_dir
        elif task == "evaluation_switching_activity":
            task_output_dir_path = self.dir_config.swact_out_dir
        elif task == "power_extraction":
            task_output_dir_path = self.dir_config.power_out_dir
            docker_image_name = "genial:latest"
        elif task == "synthesis_flowy":
            task_output_dir_path = self.dir_config.synth_out_dir
            launcher_func = Launcher.flowy_process_batch
            for shared_filepath in shared_files:
                if shared_filepath.name == "flowy_run_flow_config.yaml":
                    flowy_run_config_dict = yaml.safe_load(shared_filepath.open("r"))
                    break
            flowy_selection_metric = self.dir_config.args_dict.get("flowy_selection_metric")
            if flowy_selection_metric is not None:
                flowy_run_config_dict["selection_metric"] = flowy_selection_metric

            third_level_parallel = min(
                self.nb_workers, min(flowy_run_config_dict["chain_workers"], flowy_run_config_dict["chains"])
            )
            second_level_parallel = max(
                min(self.nb_workers // third_level_parallel, flowy_run_config_dict["nb_runs"]), 1
            )
            first_level_parallel = max(min(self.nb_workers // (second_level_parallel * third_level_parallel), end), 1)
            logger.info(
                f"Launching {task} with following parallelism: lvl1=={first_level_parallel} lvl2=={second_level_parallel} lvl3=={third_level_parallel}"
            )
            flowy_run_config_dict["nb_parallel_runs"] = second_level_parallel
            flowy_run_config_dict["chain_workers"] = third_level_parallel
            logger.disable("flowy")

        top_pid = str(os.getpid())

        pbar_desc = f"x{first_level_parallel}|{self.experiment_name} | Processing {task}"
        with tqdm(total=end, desc=pbar_desc) as pbar:  # Progress bar
            res_dir_queue = utils.process_pool_helper(
                func=launcher_func,
                func_args_gen=(
                    (
                        file_batch,
                        shared_files,
                        f"launch_script_{task}",
                        task_output_dir_path,
                        docker_image_name,
                        top_pid,
                        flowy_run_config_dict,
                    )
                    for idx, (file_batch) in enumerate(batched_files[:end])
                ),
                max_workers=first_level_parallel,
                pbar=pbar,
            )

        return res_dir_queue

    def _main_synthesis(self, design_path_list: list[str | Path]) -> list[Path]:
        """This function launches the synthesis task for all designs in design list."""

        # Launch Synthesis Tasks
        batched_files = [*zip(design_path_list)]

        shared_files = self.dir_config._get_shared_files(step="synthesis")

        # design_number_list = ["".join(filter(str.isdigit, str(design_path.relative_to(self.dir_config.root_output_dir)))) for design_path in design_path_list]
        # assert all([str.isdigit(design_number) for design_number in design_number_list]), "Error extracting design numbers from design_path_list"

        # Actually running all synthesis (mutliprocessing managed by the __launch_all function)
        if self.exp_config.get("synth_with_flowy", False):
            synth_res_dir_list = self.__launch_all(shared_files, batched_files, task="synthesis_flowy")
        else:
            synth_res_dir_list = self.__launch_all(shared_files, batched_files, task="synthesis")

        if not self.exp_config.get("synth_with_flowy", False):
            desc = f"x{self.nb_workers}|{self.experiment_name} | Cleaning up synthesized files"
            with tqdm(total=len(synth_res_dir_list), desc=desc) as pbar:  # Progress bar
                utils.process_pool_helper(
                    func=format_synthed_design,
                    func_args_gen=(
                        (synthed_design_path / "mydesign_yosys.v", self.dir_config.technology, False, True)
                        for synthed_design_path in synth_res_dir_list
                    ),
                    pbar=pbar,
                )

        return synth_res_dir_list

    def generate_testbench(
        self,
        synthed_design_path: str | Path | None,
        out_path: str | Path,
        design_number: str,
        top_synth_name: str,
        step: str = "swact",
    ) -> Path:
        """
        This function reads the design, finds all wires, and substitute the template with the list of wires.
        All wires related to registers are removed from the wire list before substituting.
        """

        # Initialise files and paths
        if step == "swact":
            subfolder = "evaluation"
            if self.dir_config.swact_ver != 0:
                subfolder += f"_v{self.dir_config.swact_ver}"
        elif step == "power":
            subfolder = "power_extraction"
            subfolder += f"_v{self.dir_config.swact_ver}"

        in_template_path = self.dir_config.exp_templates_dir / subfolder / "testbench.py.temp"
        if not in_template_path.exists():
            in_template_path = self.dir_config.default_templates_dir / subfolder / "testbench.py.temp"
        out_template_path = out_path / "testbench.py"
        template_lines = in_template_path.read_text()
        template = Template(template_lines)

        # The validating function
        custom_validating_function_filepath = (
            self.dir_config.exp_templates_dir
            / "testbench_generation"
            / f"validate_{self.exp_config['design_type']}.func"
        )
        default_validating_function_filepath = (
            self.dir_config.default_templates_dir
            / "testbench_generation"
            / f"validate_{self.exp_config['design_type']}.func"
        )
        if custom_validating_function_filepath.exists():
            validating_function_txt = custom_validating_function_filepath.read_text()
        else:
            validating_function_txt = default_validating_function_filepath.read_text()

        if synthed_design_path is not None:
            # The wire list
            wire_list, (in_wire_list, out_wire_list) = get_wire_list(
                synthed_design_path=synthed_design_path,
                remove_dangling_wires=self.remove_dangling_wires,
                clean_output_pin_names=False,
            )

            # Remove input and output wires from the list
            wire_list = [wire for wire in wire_list if wire not in in_wire_list + out_wire_list]
        else:
            wire_list = None

        # The encoding dicts
        genered_design_filepath = file_parsers.get_genered_design_file_path(
            self.dir_config.generation_out_dir, design_number
        )
        encodings_dict = file_parsers.extract_encodings(genered_design_filepath)
        encodings_dict_reversed = file_parsers.extract_encodings(genered_design_filepath, reverted=True)

        # The list of tests to be performed
        do_test_template_txt = (
            self.dir_config.default_templates_dir / "testbench_generation" / f"do_test.func"
        ).read_text()
        do_test_template = Template(do_test_template_txt)
        skip_tests_list = set(self.args_dict.get("skip_tests_list", []))
        if len(skip_tests_list) > 0:
            logger.warning(
                f"`skip_test_list` has been received. The following tests will be skipped: {skip_tests_list}"
            )
        do_tests_list = set(self.exp_config["do_tests_list"]) - skip_tests_list
        tests_list_text = ""
        for test_type in do_tests_list:
            test_text = do_test_template.substitute({"test_type": test_type, "nb_iter": self.exp_config["nb_iter"]})
            tests_list_text += test_text
            tests_list_text += "\n"

        bounding_values_dict = self.dir_config.get_bounding_values()

        # Substitute in testbench template
        substitute_dict = {
            "encodings_dict": str(encodings_dict),
            "encodings_dict_reversed": str(encodings_dict_reversed),
            "validating_function": validating_function_txt,
            "do_tests_list": tests_list_text,
        }
        if wire_list is not None:
            substitute_dict.update({"wire_list": wire_list})

        if top_synth_name == "mydesign_top":
            substitute_dict["extra_sub_dut_name"] = ""
        elif top_synth_name == "mydesign_comb":
            # SYNTHV3: If the output design is combinational only, we need to get wires form it directly
            substitute_dict["extra_sub_dut_name"] = ".i_mydesign_comb"

        substitute_dict.update(bounding_values_dict)
        testbench_strings = str(template.substitute(substitute_dict))
        if not out_template_path.parent.exists():
            out_template_path.parent.mkdir(parents=False, exist_ok=True)
        out_template_path.write_text(testbench_strings)

        return out_template_path

    def generate_synth_wrapper(
        self, synthed_design_path: str | Path | None, out_path: str | Path, step: str = "swact"
    ) -> Path:
        """This function templates the makefile and synth_wrapper files."""

        if synthed_design_path is not None:
            # Get top_synth_name
            design_lines = file_parsers.read_lines(synthed_design_path)

            top_synth_name = None
            for line in design_lines:
                if line.startswith("module"):
                    top_synth_name = line.split(" ")[1].split("(")[0]
                    break
            if top_synth_name is not None:
                assert top_synth_name in self.synth_instance_names, (
                    f"Design Number: {synthed_design_path.parent.name} | Found: {top_synth_name} | Expected: {self.synth_instance_names}"
                )  # At this stage top_synth_name should be mydesign_top__################ (from pickling) but the number should have been removed by the synthesis script
            else:
                raise ValueError(
                    f"Design Number: {synthed_design_path.parent.name} | Could not find top_synth_name. Is the file valid?"
                )
        else:
            top_synth_name = "mydesign_top"

        substitute_dict = {"post_synth_design_top_name": top_synth_name}

        if step == "power":
            substitute_dict.update(
                {
                    "in_bitwidth": self.exp_config["input_bitwidth"],
                    "out_bitwidth": self.exp_config["output_bitwidth"],
                }
            )
            template_folder = self.dir_config.default_templates_dir / f"power_extraction_v{self.dir_config.power_ver}"
        elif step == "swact":
            template_folder = self.dir_config.exp_templates_dir / "evaluation"
        else:
            raise NotImplementedError(f"step {step} not implemented for generate_synth_wrapper")

        # Read the right template
        if top_synth_name == "mydesign_comb":
            # SYNTHV3
            in_synth_wrapper_template_name = "mydesign_synth_wrapper_from_comb.v.temp"
        elif top_synth_name == "mydesign_top":
            in_synth_wrapper_template_name = "mydesign_synth_wrapper.v.temp"

        in_synth_wrapper_template_path = template_folder / in_synth_wrapper_template_name

        out_synth_wrapper_template_path = out_path / "mydesign_synth_wrapper.v"
        template_lines = in_synth_wrapper_template_path.read_text()
        template = Template(template_lines)

        # Substitute in template
        template_substitute = template.substitute(substitute_dict)
        out_synth_wrapper_template_path.write_text(str(template_substitute))

        return out_synth_wrapper_template_path, top_synth_name

    def _main_switching_activity_evaluation(self, synth_res_dir_list: list[str | Path]) -> list[Path]:
        """This function launches the switching activity measure testbench for all designs in design list."""

        # Iterate over all designs
        testbench_path_list = []
        genered_design_path_list = []
        synthed_design_path_list = []
        synth_wrapper_path_list = []
        # design_number_list = []

        with tqdm(
            total=len(synth_res_dir_list), desc=f"x1|{self.experiment_name} | Generating TBs and Synth Wraps"
        ) as pbar:  # Progress bar
            # Generate testbenches and synthesis wrappers for all designs
            for synth_dir in synth_res_dir_list:
                if not synth_dir.is_dir():
                    logger.warning(
                        f"Skipped item {synth_dir.name} because it is not a directory in {self.dir_config.synth_out_dir}"
                    )
                    continue

                # Setup input and output paths
                synth_dir_path = Path(synth_dir)
                curr_out_dir = Path(self.dir_config.power_out_dir / synth_dir_path.name)
                curr_out_dir.mkdir(exist_ok=True)
                synthed_design_path = synth_dir_path / "mydesign_yosys.v"

                design_number = file_parsers.extract_design_number_from_path(synth_dir_path)
                genered_design_dirpath = file_parsers.get_genered_design_dir_path(
                    self.dir_config.generation_out_dir, design_number
                )

                # Generate test benches
                format_synthed_design(
                    synthed_design_path,
                    technology_name=self.dir_config.technology,
                )
                synth_wrapper_path, top_synth_name = self.generate_synth_wrapper(
                    synthed_design_path=synthed_design_path, out_path=curr_out_dir, step="swact"
                )
                testbench_path = self.generate_testbench(
                    synthed_design_path=synthed_design_path,
                    out_path=curr_out_dir,
                    design_number=design_number,
                    top_synth_name=top_synth_name,
                )

                testbench_path_list.append(testbench_path)
                genered_design_path_list.append(genered_design_dirpath)
                synthed_design_path_list.append(synthed_design_path)
                synth_wrapper_path_list.append(synth_wrapper_path)
                # design_number_list.append(design_number)

                pbar.update()

        batched_files = [
            *zip(testbench_path_list, synthed_design_path_list, genered_design_path_list, synth_wrapper_path_list)
        ]

        logger.info(f"All files for SwAct jobs have been generated (testbench, synth_wrapper)")

        shared_files = self.dir_config._get_shared_files(step="evaluation")

        logger.info(f"Now launching all SwAct jobs.")
        tests_run = set(self.dir_config.exp_config["do_tests_list"]) - set(
            self.dir_config.args_dict.get("skip_test_list", [])
        )
        logger.info(f"Tests runs: {tests_run} ...")
        return self.__launch_all(shared_files, batched_files, task="evaluation_switching_activity")

    def _main_power_extraction(self, gener_res_dir_list: list[str | Path]) -> list[Path]:
        """This function launches the power extraction for all designs in design list."""

        # Launch Power Extaction Tasks
        testbench_paths_list = []
        synth_wrapper_paths_list = []
        with tqdm(
            total=len(gener_res_dir_list), desc=f"x1|{self.experiment_name} | Generating TBs"
        ) as pbar:  # Progress bar
            # Generate testbenches for all designs
            for gener_dir in gener_res_dir_list:
                # Setup input and output paths
                gener_dir_path = Path(gener_dir)
                curr_out_dir = Path(self.dir_config.power_out_dir / gener_dir_path.parent.name)

                design_number = file_parsers.extract_design_number_from_path(gener_dir_path)

                # Generate Testbench
                testbench_path = self.generate_testbench(
                    synthed_design_path=None,
                    out_path=curr_out_dir,
                    design_number=design_number,
                    top_synth_name="mydesign_top",
                    step="power",
                )
                testbench_paths_list.append(testbench_path)

                # Generate synth wrapper
                synth_wrapper_path, top_synth_name = self.generate_synth_wrapper(
                    synthed_design_path=None, out_path=curr_out_dir, step="power"
                )
                synth_wrapper_paths_list.append(synth_wrapper_path)

                pbar.update(1)

        batched_files = [*zip(gener_res_dir_list, testbench_paths_list, synth_wrapper_paths_list)]
        logger.info(f"All files for Power jobs have been generated (testbench)")

        shared_files = self.dir_config._get_shared_files(step="power_extraction")

        logger.info(f"Now launching all Power Extraction jobs.")
        tests_run = set(self.dir_config.exp_config["do_tests_list"]) - set(
            self.dir_config.args_dict.get("skip_test_list", [])
        )
        logger.info(f"Tests runs: {tests_run} ...")
        power_paths = self.__launch_all(shared_files, batched_files, task="power_extraction")

        # Do clean the synthed design
        for folder in power_paths:
            filepath = folder / "mydesign_yosys.v"
            if filepath.exists():
                format_synthed_design(
                    filepath,
                    technology_name=self.dir_config.technology,
                )

    def _get_dir_list(self, step: str) -> list[Path]:
        """This function returns the list of existing directories for the specified step."""

        assert step in ["synth", "swact"]

        if step == "synth":
            return file_parsers.get_list_of_synth_designs_dirpath(self.dir_config)
        elif step == "swact":
            return file_parsers.get_list_of_swact_designs_dirpath(self.dir_config)
        elif step == "gener":
            # TODO: same thing as synth dirpath but for gener paths ...
            return file_parsers.get_list_of_gener_designs_dirpath(self.dir_config)

    # def filter_toswact_dir_list(self, synth_dir_list:list[Path]) -> list[Path]:
    #     """ This function filters out all synthesized designs that should not be tested. """

    #     init_len = len(synth_dir_list)
    #     logger.info(f"Filtering out designs already tested ...")

    #     if self.args_dict.get("skip_already_swact", False):
    #         # Find all designs that have already been tested
    #         tested_design_number_list =  file_parsers.get_list_of_swact_designs_number(self.dir_config)

    #         # Remove them from the list of synthesized directories
    #         _synth_dir_list = []
    #         for synth_dir_path in synth_dir_list:
    #             if not "".join(filter(str.isdigit, synth_dir_path.name)) in tested_design_number_list:
    #                 _synth_dir_list.append(synth_dir_path)

    #         final_len = len(_synth_dir_list)

    #         logger.opt(colors=True).info(f"Removed <red>{init_len - final_len}</red> from list of synthesized designs to be tested")
    #         logger.opt(colors=True).info(f"Remaining <green>{final_len}</green> designs will be tested")

    #     else:
    #         _synth_dir_list = synth_dir_list

    #     # If design list is specified, test only those designs
    #     if self.user_design_number_list is not None:
    #         _user_synth_dir_list = []
    #         for design_number in self.user_design_number_list:
    #             for synth_dir_path in _synth_dir_list:
    #                 if design_number in synth_dir_path.name:
    #                     _user_synth_dir_list.append(synth_dir_path)
    #         _synth_dir_list = _user_synth_dir_list
    #         logger.info(f"List of designs to test reduced to user specified ones. Found {len(_synth_dir_list)} designs to synthesized.")

    #     return _synth_dir_list

    def filter_todo_pathlist(self, todo_design_pathlist: list[Path], step: str) -> list[Path]:
        """
        This function filters out all generated designs that should not be synthesized or swacted.
        Args:
            todo_designs_paths_list (list[Path]): list of generated designs to synthesize
            step (str): whether to filter synth or swact filepaths
        Returns:
            list[Path]: list of generated designs to synthesize
        """

        assert step in ["synth", "swact"]

        init_len = len(todo_design_pathlist)
        # logger.info(todo_design_pathlist)
        _todo_design_pathlist = []
        logger.info(f"Filtering out designs already {step}ed ...")
        if self.args_dict.get(f"skip_already_{step}", False):
            # Get list of valid designs
            if step == "synth":
                done_design_number_list = file_parsers.get_list_of_synth_designs_number(self.dir_config)
            elif step == "swact":
                done_design_number_list = file_parsers.get_list_of_swact_designs_number(self.dir_config)

            # Remove them from the list of generated directories
            todo_design_numbers = [file_parsers.extract_design_number_from_path(path) for path in todo_design_pathlist]
            todo_design_numbers_set = set(todo_design_numbers)
            done_design_numbers_set = set(done_design_number_list)
            # logger.info()

            todo_design_numbers_set = todo_design_numbers_set - done_design_numbers_set
            for path in todo_design_pathlist:
                design_number = file_parsers.extract_design_number_from_path(path)
                if design_number in todo_design_numbers_set:
                    if step == "synth":
                        _todo_design_pathlist.append(path.parent)
                    else:
                        _todo_design_pathlist.append(path)
            # logger.info(f"TODO design file paths")
            # logger.info(_todo_design_pathlist)
            # _todo_design_pathlist = file_parsers.filter_design_path_list(todo_design_pathlist, synth_design_number_list, mode="exclude")

            final_len = len(_todo_design_pathlist)

            logger.opt(colors=True).info(
                f"Removed <red>{init_len - final_len}</red> from list of generated designs to be {step}ed"
            )
            logger.opt(colors=True).info(f"Remaining <green>{final_len}</green> designs will be {step}ed")

        else:
            _todo_design_pathlist = todo_design_pathlist

        # If design list is specified, synthesize only those designs
        if self.user_design_number_list is not None:
            _todo_design_pathlist = file_parsers.filter_design_path_list(
                _todo_design_pathlist, self.user_design_number_list, mode="include"
            )
            logger.info(
                f"List of designs to synthesize reduced to user specified ones. Found {len(_todo_design_pathlist)} designs to {step}ed."
            )

        return _todo_design_pathlist

    def smart_order_todo_design_path_list_with_previous_knowledge(
        self, todo_design_df: pd.DataFrame, target_metric_name: str = "nb_transistors"
    ) -> pd.DataFrame:
        """
        This function sorts the list of design to be done based on the target_metric for the same designs of a previous experiment.
        By default, it sort them based on the number of transistors in another syntesis version.
        Warning! That function expects the design numbers to match their respective encodings. If the encodings differ, this will work be the results will not make sense.
        Return:
            - Sorted `todo_design_df`
        """

        logger.info(f"Sorting with uniform sampling the list of designs to do based on metric {target_metric_name} ...")

        # Get the metric df {"design_number", "target_metric_name"}
        if target_metric_name == "nb_transistors":
            metric_df = pd.read_parquet(self.other_synth_dir_config.analysis_out_dir / "synth_analysis.db.pqt")

        # Get the bins for each design number
        todo_design_df = split_design_numbers_in_bins_on_tgt_metric(
            to_split_df=todo_design_df,
            metric_df=metric_df,
            tgt_metric_name=target_metric_name,
        )
        # Dissociate designs known in other_synth_dir from the unkonwn ones (which thus have na in their design number)
        unknown_designs = todo_design_df[todo_design_df["bins"].isna()]
        to_split_df = todo_design_df.dropna(subset=["bins"])

        # Get the list of design numbers to keep based on their bins and on the target number of designs
        selected_design_number_list = extend_list_of_selected_designs(
            selected_design_number_list=[],
            to_split_df=to_split_df,
            tgt_nb_samples=len(todo_design_df),
            random_gen=np.random.default_rng(global_vars.get("seed")),
        )

        selected_design_number_list.extend(unknown_designs["design_number"].to_list())

        # Reorder the todo_design_df based on the order of design numbers in selected_design_number_list
        todo_design_df = todo_design_df.set_index("design_number").loc[selected_design_number_list].reset_index()

        return todo_design_df

    def main(self):
        """Full launcher run, without generation."""

        start_time = time()
        logger.info(f"Starting Launcher.main ...")

        # already_swacted_design_numbers = file_parsers.get_list_of_swact_designs_number(self.dir_config)
        # logger.info(f"Reading out")
        # genered_design_numbers = file_parsers.get_list_of_gener_designs_number(self.dir_config)
        if not self.skip_synth:
            # Get the list of generated designs to synthesize
            already_synthed_design_numbers = file_parsers.get_list_of_synth_designs_number(self.dir_config)
            logger.info(f"Getting the list of designs to synthesize ... ")
            tosynth_design_df: pd.DataFrame
            if self.args_dict.get("force", False):
                tosynth_design_df = file_parsers.get_list_of_gener_designs_dirpath(self.dir_config, get_df=True)
            else:
                if self.user_design_number_list is not None:
                    tosynth_design_df = file_parsers.get_list_of_gener_designs_dirpath(
                        self.dir_config,
                        filter_design_numbers=set(self.user_design_number_list),
                        filter_mode="include",
                        get_df=True,
                    )
                else:
                    tosynth_design_df = file_parsers.get_list_of_gener_designs_dirpath(
                        self.dir_config,
                        filter_design_numbers=already_synthed_design_numbers,
                        filter_mode="exclude",
                        get_df=True,
                    )

            # If we already have some knowledge about preexisting synthesized designs, use this knowledge to order the designs to test
            if self.other_synth_dir_config is not None:
                tosynth_design_df = self.smart_order_todo_design_path_list_with_previous_knowledge(
                    tosynth_design_df,
                    "nb_transistors",
                )
            else:
                logger.info(f"No other synth path specified, synthesizing design in random order.")

            if len(tosynth_design_df) > 0:
                tosynth_design_df["hdl_path"] = tosynth_design_df["design_path"].map(lambda x: x / "hdl")
                tosynth_design_list = tosynth_design_df["hdl_path"].tolist()
            else:
                tosynth_design_list = []

            # Run all synthesis jobs
            logger.info(f"Starting design synthesis of {len(tosynth_design_list)} designs")
            synthed_dir_list = self._main_synthesis(tosynth_design_list)
            logger.info(f"Finished synthesis step")
            logger.info(f"All files written in {self.dir_config.synth_out_dir}")

        else:
            logger.info(f"Skipped design synthesis loop")
            synthed_dir_list = []

        # Run all switching activity analysis jobs
        if not self.skip_swact:
            already_swacted_design_numbers = file_parsers.get_list_of_swact_designs_number(self.dir_config)

            # Get the list of swacted designs to synthesize
            logger.info(f"Getting the list of designs to swact ... ")
            if len(synthed_dir_list) == 0:
                if self.args_dict.get("force"):
                    toswact_dir_list = file_parsers.get_list_of_synth_designs_dirpath(self.dir_config)
                else:
                    toswact_dir_list = file_parsers.get_list_of_synth_designs_dirpath(
                        self.dir_config, filter_design_numbers=already_swacted_design_numbers, filter_mode="exclude"
                    )
            else:
                toswact_dir_list = synthed_dir_list

            # toswact_dir_list = self.filter_todo_pathlist(synthed_dir_list, step="swact")
            logger.info(f"Starting switching activity measurement of {len(toswact_dir_list)} designs")
            swacted_dir_list = self._main_switching_activity_evaluation(toswact_dir_list)
            logger.info(f"Finished switching activity measurement loop")
            logger.info(f"All files written in {self.dir_config.swact_out_dir}")
        else:
            logger.info(f"Skipped switching activity measurement loop")
            swacted_dir_list = []

        # Run all power extraction jobs
        if not self.skip_power:
            already_powered_design_numbers = file_parsers.get_list_of_power_designs_number(self.dir_config)

            # Get the list of designs to extract the power from
            # to_powered_design_numbers = list()

            logger.info(f"Getting the list of designs to power ... ")
            topower_design_df: pd.DataFrame
            if self.args_dict.get("force", False):
                topower_design_df = file_parsers.get_list_of_gener_designs_dirpath(self.dir_config, get_df=True)
            else:
                if self.user_design_number_list is not None:
                    topower_design_df = file_parsers.get_list_of_gener_designs_dirpath(
                        self.dir_config,
                        filter_design_numbers=set(self.user_design_number_list),
                        filter_mode="include",
                        get_df=True,
                    )
                else:
                    topower_design_df = file_parsers.get_list_of_gener_designs_dirpath(
                        self.dir_config,
                        filter_design_numbers=already_powered_design_numbers,
                        filter_mode="exclude",
                        get_df=True,
                    )

            if len(topower_design_df) > 0:
                topower_design_list = [path / "hdl" for path in topower_design_df["design_path"].to_list()]
            else:
                topower_design_list = []

            logger.info(f"Starting power extraction measurement of {len(topower_design_list)} designs")
            powered_dir_list = self._main_power_extraction(topower_design_list)
            logger.info(f"Finished power extraction measurement loop")
            logger.info(f"All files written in {self.dir_config.power_out_dir}")
        else:
            logger.info(f"Skipped power extraction measurement loop")
            powered_dir_list = []

        end_time = time()
        logger.info(f"Launcher.main done in {(end_time - start_time) / 60 / 60:.2f}h.\n")

        return [], synthed_dir_list, swacted_dir_list, powered_dir_list

    @staticmethod
    def parse_args() -> dict[str, Any]:
        default_args_dict = get_default_parser()

        arg_parser = ArgumentParser(description="Run the launcher")
        arg_parser.add_argument(
            "--only_gener", action="store_true", help="Do only the generation of design verilog files"
        )
        # arg_parser.add_argument("--skip_gener", action="store_true", help="Skip the generation of design verilog files")
        arg_parser.add_argument(
            "--do_gener",
            action="store_true",
            help="Do the generation of design verilog files. OFF by default to avoid overwritting existing generated designs.",
        )
        arg_parser.add_argument(
            "--nb_new_designs",
            type=int,
            default=None,
            help="Nb of designs to generate. If set, it will override the do_gener parameter.",
        )
        arg_parser.add_argument(
            "--skip_already_synth", action="store_true", help="Do not synthesize already existing designs"
        )
        arg_parser.add_argument("--skip_already_swact", action="store_true", help="Do not test already tested designs")
        arg_parser.add_argument(
            "--skip_already_power",
            action="store_true",
            help="Do not do power extraction of already power extracted designs ",
        )
        arg_parser.add_argument("--force", action="store_true", help="Enforce re-doing already done tasks")
        arg_parser.add_argument(
            "--restart", action="store_true", help="Simplify command line interface for restarting a job"
        )
        arg_parser.add_argument(
            "--skip_tests_list",
            nargs="*",
            type=str,
            default=[],
            help="Tests that should be skipped during launcher swact operations.",
        )
        args = arg_parser.parse_known_args()

        args_dict = vars(args[0])
        args_dict.update(default_args_dict)

        if args_dict["restart"]:
            args_dict["do_gener"] = False
            args_dict["skip_already_synth"] = True
            args_dict["skip_already_swact"] = True
            args_dict["skip_already_power"] = True

        if args_dict["nb_new_designs"] is not None:
            args_dict["do_gener"] = True

        return args_dict

    def main_standalone(self):
        """Full launcher run, with generation (if do_gener is set)."""

        start_time = time()

        res_dict = {"start_time": start_time}

        # Run the design generator
        if self.args_dict.get("do_gener", True) and self.args_dict.get("design_number_list", None) is None:
            # Initialize the design generator
            design_generator = DesignGenerator(dir_config=self.dir_config)
            origin_genered_design_list = design_generator.main(nb_to_generate=self.args_dict.get("nb_new_designs"))
            res_dict.update({"origin_genered_design_list": origin_genered_design_list})
        else:
            # origin_genered_design_list = []
            logger.warning(f"Generation of design verilog files have been skipped")

        # Run the task launcher (synthesis and switching activity measurement)
        if not self.args_dict.get("only_gener", False):
            genered_design_list, synthed_dir_list, swacted_dir_list, powered_dir_list = self.main()
            updated_entries = {
                "genered_design_list": genered_design_list,
                "synthed_dir_list": synthed_dir_list,
                "swacted_dir_list": swacted_dir_list,
                "powered_dir_list": powered_dir_list,
            }
            res_dict.update(updated_entries)
        else:
            logger.info(f"Launcher has been skipped since `only_gener` was received")

        end_time = time()
        res_dict.update({"end_time": end_time})

        if self.current_iter_nb == 0:
            self.build_report(res_dict=res_dict)

        logger.info(f"Launcher.main_standalone done in {(end_time - start_time) / 60 / 60:.2f}h.\n")

        return res_dict

    def build_report(self, res_dict: dict[str:Any]):
        """ """

        report_lines = [
            f"Start Time = {datetime.fromtimestamp(res_dict['start_time'])}\n",
        ]
        report_lines += [
            f"End Time = {datetime.fromtimestamp(res_dict['end_time'])}\n",
        ]
        report_lines += [
            f"Duration = {datetime.fromtimestamp(res_dict['end_time'])}\n",
        ]
        report_lines += [
            f"# Workers = {self.nb_workers}\n",
        ]
        # report_lines += [f"# Generated Designs = {len(res_dict["origin_genered_design_list"])}\n",]
        try:
            report_lines += [
                f"# Designs to Synthesize = {len(res_dict['genered_design_list'])}\n",
            ]
        except (KeyError, TypeError):
            pass
        try:
            report_lines += [
                f"# Designs to Test = {len(res_dict['synthed_dir_list'])}\n",
            ]
        except (KeyError, TypeError):
            pass
        try:
            report_lines += [
                f"# Successfully Tested Designs = {len(res_dict['swacted_dir_list'])}\n",
            ]
        except (KeyError, TypeError):
            pass
        report_lines += [
            f"\n",
        ]

        report_file_path = self.dir_config.root_output_dir / "launcher_run_from_scratch_info.txt"
        with open(report_file_path, "a") as report_file:
            report_file.writelines(report_lines)
            logger.info(f"Launcher report stored in:")
            logger.info(f"{report_file_path}")

        return report_lines

    def switch_to_iter_mode(self, config_dict: dict[str, Any]) -> None:
        """Update the launcher state so that it runs in iteration mode."""
        self.args_dict["restart"] = True
        self.args_dict["do_gener"] = False
        self.args_dict["only_gener"] = False
        self.args_dict["skip_already_synth"] = True
        self.args_dict["skip_already_swact"] = True

    def prepare_next_iter(self) -> bool:
        """
        Prepare the next iteration nby updating the list of designs todo.
        Returns:
            True if something went wrong.
            False otherwise.
        """
        return False


def main_cli() -> None:
    try:
        launcher = Launcher()
        launcher.main_standalone()

        status = "Success"
        error_msg = ""

    except Exception:
        status = "Failed"
        error_msg = traceback.format_exc()
        logger.error(error_msg)

    # launcher.send_email(
    #     config_dict=launcher.args_dict,
    #     start_time=start_time,
    #     status=status,
    #     error_message=error_msg,
    #     calling_module="Launcher",
    #     root_output_dir=launcher.dir_config.root_output_dir,
    # )

    if status == "Failed":
        logger.error(error_msg)
    logger.info(f"Launcher's `main_cli` exited with status:{status}.")


if __name__ == "__main__":
    main_cli()
