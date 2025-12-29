from dataclasses import asdict, dataclass
import os
import subprocess
from subprocess import CompletedProcess
from time import time, sleep

import yaml
from loguru import logger

import shutil
from pathlib import Path
import traceback
import sys

from genial.config.config_dir import ConfigDir

from string import Template
import re

import tempfile

from genial.globals import global_vars
from datetime import datetime

from genial.experiment.file_parsers import get_list_of_gener_designs_number, get_list_of_synth_designs_number

# from genial.utils.slurm import SlurmGPUFinder


from genial.experiment.file_parsers import _get_list_of_valid_designs
from genial.experiment.task_analyzer import Analyzer
from types import SimpleNamespace


global_vars["is_slurm"] = False


class SlurmDispatcher:
    __valid_tasks__ = [
        "generate",
        "launch",
        "analyze",
        "train",
        "recommend",
        "merge",
        "clean",
    ]

    __valid_nodes__ = [
        "aisrv01",
        # "aisrv02",
        "aisrv03",
        # "aime01",
        # "aime02",
        # "aime03",
        # "epyc01",
        # "epyc02",
    ]

    __valid_work_dirpath__ = [
        "/netscratch/aisrv01",
        "/netscratch/aisrv02",
        "/netscratch/aisrv03",
        "/netscratch/epyc01",
        "/netscratch/epyc02",
        "/netscratch/aime01",
        "/netscratch/aime02",
        "/netscratch/aime03",
    ]

    __partition__ = {
        # "generate": "AI-CPU,Zen3",
        # "launch": "AI-CPU,Zen3",
        # "analyze": "AI-CPU,Zen3",
        # "train": "AI-CPU",
        # "recommend": "AI-CPU",
        # "merge": "AI-CPU,Zen3",
        # "clean": "AI-CPU,Zen3",

        "generate": "AI-CPU,VNL_GPU,Zen3",
        "launch": "AI-CPU,VNL_GPU,Zen3",
        "analyze": "AI-CPU,VNL_GPU,Zen3",
        "train": "AI-CPU,VNL_GPU,Zen3",
        "recommend": "AI-CPU,VNL_GPU,Zen3",
        "merge": "AI-CPU,VNL_GPU,Zen3",
        "clean": "AI-CPU,VNL_GPU,Zen3",
    }

    __mem_per_cpu__ = {
        "AI-CPU,Zen3": "3G",
        "VNL_GPU": "2G",
    }

    __nodelist__ = {
        "generate": "aisrv01,,aisrv02,aisrv03,aime01,aime02,aime03,epyc01,epyc02",
        # "launch": "aisrv01,aisrv02,aisrv03,aime01,aime02,aime03,epyc01,epyc02",
        # "launch": "aisrv01,aisrv03,aime01,aime02,aime03,epyc01,epyc02",
        # "launch": "aisrv01,aisrv02,aisrv03",
        "launch": "aisrv01,aisrv03",
        "analyze": "aisrv01,aisrv02,aisrv03,aime01,aime02,aime03,epyc01,epyc02",
        "train": "aime01,aime02,aime03",
        "recommend": "aime01,aime02,aime03",
        "merge": "aisrv01,aisrv02,aisrv03",
    }

    __can_multi_node__ = {
        "generate": False,
        "launch": True,
        "analyze": False,
        "train": False,
        "recommend": False,
    }

    @staticmethod
    def check_slurm_setup(**kwargs):
        """
        Check if the slurm setup is correct.
        """
        global_vars["is_slurm"] = kwargs.get("is_slurm", False)
        if global_vars["is_slurm"]:
            logger.warning("Slurm setup is enabled.")

        # Do the right checks
        work_dirpath = Path(os.environ.get("WORK_DIR"))
        if str(work_dirpath.parents[-3]) not in SlurmDispatcher.__valid_work_dirpath__:
            logger.error(
                f"Please make sure that the `WORK_DIR` environement variable is set to a valid scratch disk before pursuing."
                f"\nReceived: {work_dirpath}"
                f"List of valid scratch disks:"
                f"\n{SlurmDispatcher.__valid_work_dirpath__}"
            )
            raise EnvironmentError(
                f"Please make sure that the `WORK_DIR` environement variable is set to a valid scratch disk before pursuing."
                f"List of valid scratch disks:"
                f"{SlurmDispatcher.__valid_work_dirpath__}"
            )

        logdirpath = Path(f"/home/{os.environ.get('USER')}/slurm_logs/genial/sbatch_info")
        if not logdirpath.exists():
            logdirpath.mkdir(parents=True, exist_ok=True)
        logdirpath = Path(f"/home/{os.environ.get('USER')}/slurm_logs/genial/sbatch_error")
        if not logdirpath.exists():
            logdirpath.mkdir(parents=True, exist_ok=True)

        tmp_scripts_dir = Path(os.environ.get("HOME") + "/tmp_genial_slurm_scripts")
        if not tmp_scripts_dir.exists():
            tmp_scripts_dir.mkdir(parents=True, exist_ok=True)

        global_vars["keep_not_valid"] = True

    @staticmethod
    def generate_sbatch_args(task: str, nb_workers: int):
        _nb_workers = nb_workers
        # node = None
        device_ids = None
        time = "3:00:00"

        print(f"=========================== {task}")
        if task == "train" or task == "recommend":
            _nb_workers = 16
            # checker = SlurmGPUFinder(partition="VNL_GPU")  # Change 'gpu' if needed
            # available_gpus = checker.find_node_with_free_gpu()
            device_ids = [0]
            time = "48:00:00"

        if task == "launch" or task == "clean":
            time = "2:00:00"
            _nb_workers = 24

        cpus_per_task = str(int(_nb_workers * 1.2))

        cmd_prefix_list = [
            f"--job-name={task}_genial_flowy",
            f"--time={time}",
            # "--ntasks=1",
            f"--mem-per-cpu=2G",
            # "--reservation=ai-team",
            "--partition=" + SlurmDispatcher.__partition__[task],
            f"--cpus-per-task={cpus_per_task}",
            # "--nodelist=" + SlurmDispatcher.__nodelist__[task],
        ]

        if task in ["analyze", "merge", "train", "recommend"]:
            cmd_prefix_list += ["--nodelist=aisrv01,aisrv02,aisrv03,aime01,aime02,aime03,epyc01,epyc02"]
        elif task == "clean":
            # For cleaning, submit one job per available node to speed things up.
            # Filter out nodes that are not available to avoid infinite waits.
            available_nodes = SlurmDispatcher.get_available_nodes(SlurmDispatcher.__valid_nodes__)
            if not available_nodes:
                logger.warning("No usable nodes detected for clean task; skipping clean submissions.")
            else:
                if set(available_nodes) != set(SlurmDispatcher.__valid_nodes__):
                    skipped = list(set(SlurmDispatcher.__valid_nodes__) - set(available_nodes))
                    logger.warning(f"Skipping unavailable nodes for clean task: {sorted(skipped)}")
            cmd_prefix_list = [cmd_prefix_list + [f"--nodelist={node}"] for node in available_nodes]

        # if task == "launch":
        #     cmd_prefix_list += (f"--cpus-per-task={str(int(24 * 1.5))}",)  # Use a margin of 1.5 - just to make sure.
        # else:
        #     cmd_prefix_list += (
        #         f"--cpus-per-task={}",
        # )  # Use a margin of 1.5 - jst to make sure.

        # if SlurmDispatcher.__partition__[task] == "VNL_GPU":
        # cmd_prefix_list += ["--exclude=aime02,aime01"]
        # if SlurmDispatcher.__partition__[task] == "AI-CPU,Zen3" and not task == "clean":
        #     # cmd_prefix_list += ["--exclude=epyc01,epyc02,aisrv01,aisrv02"]
        #     pass

        # if node is not None:
        # cmd_prefix_list.append(f"--nodelist={node}")
        if device_ids is not None:
            cmd_prefix_list.append(f"--gres=gpu:1")

        return cmd_prefix_list, device_ids

    @staticmethod
    def get_available_nodes(candidate_nodes: list[str]) -> list[str]:
        """Return nodes from candidate_nodes that appear available in Slurm.

        A node is considered available if its state starts with one of:
        'idle', 'mix', 'alloc', 'comp' (completing). Nodes in 'down', 'drain',
        'maint', 'fail', 'unkn' states are excluded.
        If Slurm commands are unavailable, fall back to the candidates as-is.
        """
        try:
            # Query node states once and parse
            result = subprocess.run(
                ["sinfo", "-h", "-N", "-o", "%N|%t"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            states = {}
            for line in result.stdout.strip().splitlines():
                if not line:
                    continue
                try:
                    name, state = line.split("|", 1)
                except ValueError:
                    continue
                states[name.strip()] = state.strip().lower()

            ok_prefixes = ("idle", "mix", "alloc", "comp")
            bad_prefixes = ("down", "drain", "drng", "maint", "fail", "unkn")

            usable = []
            for node in candidate_nodes:
                st = states.get(node)
                if st is None:
                    # If node is unknown to sinfo, better to skip than to block
                    logger.warning(f"Node '{node}' missing from sinfo; skipping for clean task.")
                    continue
                if st.startswith(bad_prefixes):
                    continue
                if st.startswith(ok_prefixes):
                    usable.append(node)
            return usable
        except Exception as e:
            # On any error (e.g., not on a Slurm login node), return provided list
            logger.warning(f"Unable to query node availability via sinfo ({e}); using candidate nodes as-is.")
            return list(candidate_nodes)

    @staticmethod
    def generate_slurm_launch_script(template_filepath: str) -> str:
        """
        Generate a slurm launch script from a template file by substituting
        only the required keys found in the template.

        Args:
            template_filepath (str): Path to the SLURM script template file.
            **kwargs: Keys and values to substitute in the template.

        Returns:
            str: The generated script as a string with substitutions applied.
        """
        with open(template_filepath, "r", encoding="utf-8") as f:
            template_content = f.read()

        template = Template(template_content)
        return template

    @staticmethod
    def run_cmd_slurm(cmd: str | list[str], task: str, nb_workers: int = 1, is_dry_run: bool = False):
        """
        Generate and submit a SLURM script for the given command.
        Args:
            cmd (str): The shell command to run.
            slurm_template_path (str): Path to the SLURM template file (expects $cmd).
            slurm_mode (str): "sbatch" (default) or "srun".
            **kwargs: Additional keys for the template.
        Returns:
            subprocess.CompletedProcess: The sbatch/srun submission result.
        """

        if isinstance(cmd, str):
            _cmd_list = [cmd]
        else:
            _cmd_list = cmd

        # Instantiate the sbatch args list
        sbatch_args, device_ids = SlurmDispatcher.generate_sbatch_args(task, nb_workers=nb_workers)

        if task == "clean":
            # One clean command per sbatch args (per available node)
            _cmd_list = [cmd] * len(sbatch_args)
            assert len(sbatch_args) == len(_cmd_list), (
                f"Number of commands and sbatch args must match for clean task. {len(sbatch_args)} != {len(_cmd_list)}"
            )
            template = None
        else:
            # Generate the SLURM script from template
            slurm_template_path = (
                Path(os.environ.get("SRC_DIR")) / "scripts/slurm_scripts/sbatch_dispatch_slurm_temp.sh"
            )
            template = SlurmDispatcher.generate_slurm_launch_script(slurm_template_path)

        # Generate a script for all commands
        if template is not None:
            script_pathlist = dict()
            for _cmd in _cmd_list:
                script_object = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".slurm", delete=False, dir=os.environ.get("HOME") + "/tmp_genial_slurm_scripts"
                )
                os.chmod(script_object.name, 0o777)

                if device_ids is not None:
                    if "--device" in _cmd:
                        # Find the device numebr specified (search for 'device <nb>')
                        device_nb = re.search(r"--device (\d+)", _cmd).group(1)
                        # Replace the device number with the one from the list
                        _cmd = _cmd.replace(
                            f"--device {device_nb}", f"--device {device_ids[-1]}"
                        )  # TODO: make this compatible with multi cmd in parallel ... (useless for now)

                script_content = template.substitute(
                    {
                        "cmd": _cmd,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    }
                )
                script_object.write(script_content)
                # Force flush to disk
                script_object.flush()
                os.fsync(script_object.fileno())
                script_pathlist[script_object.name] = (script_object, _cmd)

            # Submit all scripts
            any_errors = SlurmDispatcher.dispatch_all_scripts(
                script_pathlist=script_pathlist.keys(),
                sbatch_args=sbatch_args,
                max_retries=2,
                poll_interval=15,
                is_dry_run=is_dry_run,
                cancel_on_unavailable=True,
            )

            for script_path, (script_object, _cmd) in script_pathlist.items():
                script_object.close()

            if len(any_errors) > 0:
                logger.error(traceback.format_exc())
                logger.error("sbatch args were: {}".format(sbatch_args))
                raise RuntimeError("Error submitting the following commands: {}".format(any_errors))

        else:
            # Submit all commands directly
            # For 'clean', we do not want to endlessly retry jobs on unavailable nodes.
            # Cancel unschedulable jobs immediately and do not retry.
            any_errors = SlurmDispatcher.dispatch_all_jobs(
                cmd_list=_cmd_list,
                sbatch_args_list=sbatch_args,
                is_dry_run=is_dry_run,
                max_retries=0 if task == "clean" else 2,
                cancel_on_unavailable=True,
            )

        return any_errors

    @staticmethod
    def submit_script(script_path, extra_args: list[str] = None, is_dry_run=False):
        cmd = ["sbatch"]
        if extra_args:
            cmd += extra_args
        cmd.append(script_path)

        # Update Design Numbers in the Script if they Are Specified
        SlurmDispatcher.update_design_number_list(script_path)

        if is_dry_run:
            logger.info("Would submit command: {}".format(" ".join(cmd)))
            job_id = "dummy"
        else:
            try:
                logger.info(cmd)
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command {cmd} returned non-zero exit status {e.returncode}.")
                logger.error(f"Command output: --->\n{e.stdout}\n<---")
                logger.error(f"Command error: --->\n{e.stderr}\n<---")
                raise e

            job_id = int(result.stdout.strip().split()[-1])

        return job_id

    @staticmethod
    def submit_job(cmd: str, sbatch_args: list[str], is_dry_run=False):
        _cmd = (
                ["sbatch"] +
                sbatch_args  +
                # ["--reservation=ai-team"] +
                [cmd]
        )

        if is_dry_run:
            logger.info("Would submit command: {}".format(" ".join(_cmd)))
            job_id = "dummy"
        else:
            try:
                logger.info(_cmd)
                result = subprocess.run(_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command {_cmd} returned non-zero exit status {e.returncode}.")
                logger.error(f"Command output: --->\n{e.stdout}\n<---")
                logger.error(f"Command error: --->\n{e.stderr}\n<---")
                raise e

            job_id = int(result.stdout.strip().split()[-1])

        return job_id

    @staticmethod
    def get_job_status(job_id):
        # sacct output: JobID State
        cmd = ["sacct", "-j", str(job_id), "--format=JobID,State", "--noheader", "--parsable2"]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            # Parse first line for main job
            for line in result.stdout.strip().splitlines():
                if line.startswith(str(job_id)):
                    raw_state = line.split("|")[1].strip()
                    # Canonicalize state (e.g., 'CANCELLED by 10102' -> 'CANCELLED')
                    state_base = raw_state.split()[0].upper()
                    # Map extended failure-like states to FAILED
                    if state_base in {"OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}:
                        state_base = "FAILED"
                    return state_base
        except Exception:
            return None  # If sacct not available yet, e.g. right after submission
        return None

    @staticmethod
    def get_squeue_state_and_reason(job_id):
        """Return (state, reason) from squeue for a job, or (None, None).

        Uses: squeue -j <id> -h -o "%T|%R"
        """
        cmd = ["squeue", "-j", str(job_id), "-h", "-o", "%T|%R"]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            line = result.stdout.strip().splitlines()
            if not line:
                return None, None
            line = line[0]
            parts = line.split("|", 1)
            state = parts[0].strip().upper() if len(parts) > 0 else None
            reason = parts[1].strip() if len(parts) > 1 else None
            return state, reason
        except Exception:
            return None, None

    @staticmethod
    def cancel_job(job_id):
        try:
            subprocess.run(["scancel", str(job_id)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            pass

    @staticmethod
    def dispatch_all_jobs(
        cmd_list: list[list[str]],
        sbatch_args_list: list[list[str]],
        is_dry_run: bool = False,
        max_retries: int = 2,
        cancel_on_unavailable: bool = True,
    ):
        jobs = {
            job_index: {
                "tries": 0,
                "job_id": None,
                "status": None,
                "cmd": cmd,
                "sbatch_args": sbatch_args_list[job_index],
            }
            for job_index, cmd in enumerate(cmd_list)
        }
        pending = set(range(len(cmd_list)))
        any_errors = []

        while pending:
            for job_index in list(pending):
                # If not yet submitted or needs retry
                if jobs[job_index]["job_id"] is None or (
                    jobs[job_index]["status"] in ["FAILED", "CANCELLED", "TIMEOUT"]
                    and jobs[job_index]["tries"] < max_retries
                ):
                    jobs[job_index]["job_id"] = SlurmDispatcher.submit_job(
                        jobs[job_index]["cmd"], jobs[job_index]["sbatch_args"], is_dry_run
                    )
                    jobs[job_index]["tries"] += 1
                    jobs[job_index]["status"] = "SUBMITTED"
                    logger.info(
                        f"Submitted job index {job_index} as job {jobs[job_index]['job_id']} (try {jobs[job_index]['tries']})"
                    )
                    sleep(1.0)

            # Check all job status
            for job_index in pending.copy():
                job_id = jobs[job_index]["job_id"]
                if job_id is not None:
                    # Check if we are in a dry mode
                    if job_id == "dummy":
                        pending.remove(job_index)
                        continue
                    status = SlurmDispatcher.get_job_status(job_id)
                    # If stuck pending with an unschedulable reason, cancel to avoid hanging forever
                    if status == "PENDING" and cancel_on_unavailable:
                        sq_state, sq_reason = SlurmDispatcher.get_squeue_state_and_reason(job_id)
                        if sq_state == "PENDING" and sq_reason is not None:
                            reason_lc = sq_reason.lower()
                            if "reqnodenotavail" in reason_lc or "unavailablenodes" in reason_lc:
                                pass
                                # logger.warning(
                                #     f"Job {job_id} appears unschedulable (reason: {sq_reason}). Cancelling to avoid hang."
                                # )
                                # SlurmDispatcher.cancel_job(job_id)
                                # jobs[job_index]["status"] = "CANCELLED"
                                # # Let the normal failure/cancel flow handle retries/termination
                                # continue
                    if status and status not in ["PENDING", "RUNNING"]:
                        jobs[job_index]["status"] = status
                        logger.info(f"Job {job_id} ({job_index}) finished with status: {status}")
                        if status in ["COMPLETED"]:
                            pending.remove(job_index)
                        elif status in ["FAILED", "CANCELLED", "TIMEOUT"] and jobs[job_index]["tries"] >= max_retries:
                            logger.info(
                                f"Job {job_id} ({job_index}) failed after {jobs[job_index]['tries']} tries. Giving up."
                            )
                            pending.remove(job_index)
                            any_errors.append(job_index)

        logger.info("All jobs done!")
        return any_errors

    @staticmethod
    def dispatch_all_scripts(
        script_pathlist: list[str],
        sbatch_args: list[str],
        max_retries=2,
        poll_interval=0.05,
        is_dry_run: bool = False,
        cancel_on_unavailable: bool = True,
    ):
        jobs = {script: {"tries": 0, "job_id": None, "status": None} for script in script_pathlist}
        pending = set(script_pathlist)
        any_errors = []

        very_first = True
        while pending:
            for script in list(pending):
                # If not yet submitted or needs retry
                if jobs[script]["job_id"] is None or (
                    jobs[script]["status"] in ["FAILED", "CANCELLED", "TIMEOUT"] and jobs[script]["tries"] < max_retries
                ):
                    jobs[script]["job_id"] = SlurmDispatcher.submit_script(script, sbatch_args, is_dry_run)
                    jobs[script]["tries"] += 1
                    jobs[script]["status"] = "SUBMITTED"
                    logger.info(f"Submitted {script} as job {jobs[script]['job_id']} (try {jobs[script]['tries']})")
                    sleep(1.0)
                    if very_first and not is_dry_run:
                        sleep(30.0)
                        very_first = False

            # Check all job status
            for script in pending.copy():
                job_id = jobs[script]["job_id"]
                if job_id is not None:
                    # Check if we are in a dry mode
                    if job_id == "dummy":
                        pending.remove(script)
                        continue
                    status = SlurmDispatcher.get_job_status(job_id)
                    # If stuck pending with an unschedulable reason, cancel to avoid hanging forever
                    if status == "PENDING" and cancel_on_unavailable:
                        sq_state, sq_reason = SlurmDispatcher.get_squeue_state_and_reason(job_id)
                        if sq_state == "PENDING" and sq_reason is not None:
                            reason_lc = sq_reason.lower()
                            if "reqnodenotavail" in reason_lc or "unavailablenodes" in reason_lc:
                                pass
                                # logger.warning(
                                #     f"Job {job_id} appears unschedulable (reason: {sq_reason}). Cancelling to avoid hang."
                                # )
                                # SlurmDispatcher.cancel_job(job_id)
                                # jobs[script]["status"] = "CANCELLED"
                                # continue
                    if status and status not in ["PENDING", "RUNNING"]:
                        jobs[script]["status"] = status
                        logger.info(f"Job {job_id} ({script}) finished with status: {status}")
                        if status in ["COMPLETED"]:
                            pending.remove(script)
                        elif status in ["FAILED", "CANCELLED", "TIMEOUT"] and jobs[script]["tries"] >= max_retries:
                            logger.info(
                                f"Job {job_id} ({script}) failed after {jobs[script]['tries']} tries. Giving up."
                            )
                            pending.remove(script)
                            if "--job-name=launch_genial_flowy" not in sbatch_args:
                                any_errors.append(script)

                        # else: will resubmit on next loop

        logger.info("All jobs done!")
        return any_errors

    @staticmethod
    def update_design_number_list(script_path: Path):
        """Update the design number list in the script with the current design number list"""
        global_vars["keep_not_valid"] = True
        script_path = Path(script_path)
        if not script_path.exists():
            return

        with open(script_path, "r") as f:
            script = f.read()

        skipped_steps = set()
        if "--design_number_list" in script:
            for line in script.splitlines():
                if "--design_number_list" in line:
                    design_number_list = line.split("--design_number_list")[1].strip().split()

                if "--skip_" in line:
                    step_name = line.split("--skip_")[1].strip().split()[0]
                    skipped_steps.add(step_name)

                if "--experiment_name" in line:
                    experiment_name = line.split("--experiment_name")[1].strip().split()[0]

                if "--output_dir_name" in line:
                    output_dir_name = line.split("--output_dir_name")[1].strip().split()[0]

                if "--bulk_flow_dirname" in line:
                    bulk_flow_dirname = line.split("--bulk_flow_dirname")[1].strip().split()[0]
                else:
                    if "flowy" in line:
                        bulk_flow_dirname = "synth_out"
                    else:
                        bulk_flow_dirname = None

            root_dirpath = ConfigDir.get_root_dirpath(experiment_name, output_dir_name)
            dir_config = SimpleNamespace(
                root_output_dir=root_dirpath,
                bulk_flow_dirname=bulk_flow_dirname,
            )

            valid_design_numbers = {}
            for step in Analyzer.__existing_steps__:
                if step != "gener":
                    if step not in skipped_steps:
                        valid_design_numbers[step] = _get_list_of_valid_designs(
                            dir_config,
                            step,
                            return_types="numbers",
                            filter_design_numbers=design_number_list,
                            filter_mode="include",
                        )

            all_valid_design_numbers = set(design_number_list)
            for step in valid_design_numbers:
                all_valid_design_numbers = all_valid_design_numbers.intersection(set(valid_design_numbers[step]))

            todo_design_number_list = list(set(design_number_list) - all_valid_design_numbers)

            # new_script_l, new_script_r = script.split("--design_number_list")
            for dn in design_number_list:
                script = script.replace(f"{dn}", "")

            _todo_design_number_list = " ".join(todo_design_number_list)
            script = script.replace("--design_number_list", f"--design_number_list {_todo_design_number_list}")

            # Write back updated script
            with open(script_path, "w") as f:
                f.write(script)
            logger.info(f"Script {script_path} updated with new design number list: {todo_design_number_list}")

            return len(todo_design_number_list)

        else:
            return -1


def run_cmd_subprocess(cmd: str, is_dry_run: bool = False, task: str = None) -> CompletedProcess:
    #
    if task is not None and task in ["clean"]:
        cmd = cmd
    else:
        src_dir = os.environ["SRC_DIR"]
        cmd = ["source .env && cd " + src_dir + " && PYTHONPATH=" + src_dir + "/src " + cmd]

    # cmd_str = " ".join(cmd)
    logger.info("")
    logger.info(f"|| --------------------------------------------------------------------------------------")
    logger.info(f"Running command:")
    logger.info(f"{cmd}")

    start = time()
    if is_dry_run:
        logger.info(f"Running skipped")
        process = None
    else:
        try:
            process = subprocess.run(
                cmd,
                shell=True,
                check=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # To get string output instead of bytes
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Command {cmd} returned non-zero exit status {e.returncode}.")
            logger.error(f"Command output: --->\n{e.stdout}\n<---")
            logger.error(f"Command error: --->\n{e.stderr}\n<---")
            raise e

    end = time()
    logger.info(f"Done in {end - start:.2f}s.")
    logger.info(f"|| --------------------------------------------------------------------------------------")

    return process


def run_cmd(cmd: list[str] | str, task: str, nb_workers: int = 1, is_dry_run: bool = False):
    if global_vars["is_slurm"]:
        process = SlurmDispatcher.run_cmd_slurm(cmd=cmd, task=task, nb_workers=nb_workers, is_dry_run=is_dry_run)
    else:
        process = run_cmd_subprocess(cmd=cmd, is_dry_run=is_dry_run)

    return process


def clean_task(**kwargs):
    """
    Build the command to clean a launcher task and run it.
    """
    clean_up_script_path = Path(f"/home/{os.environ.get('USER')}/slurm-scripts/scripts/sbatch/sbatch_docker_cleanup.sh")
    cmd = f"{clean_up_script_path}"

    run_cmd(cmd, task="clean", nb_workers=kwargs["nb_workers"], is_dry_run=kwargs["is_dry_run"])


def launch_task(**kwargs) -> CompletedProcess:
    """
    Build the command to run a launcher task and run it.
    """

    if global_vars["is_slurm"]:
        clean_task(**kwargs)

    cmd = ["python", "src/genial/experiment/task_launcher.py"]

    # Required parameters with arguments
    for key in [
        "experiment_name",
        # "nb_workers",
        "output_dir_name",
    ]:
        cmd += ["--" + key, str(kwargs[key])]

    if global_vars["is_slurm"]:
        cmd += ["--nb_workers", str(int(24 * 1.5))]
    else:
        cmd += ["--nb_workers", str(kwargs["nb_workers"])]

    # Optional parameters with arguments
    for key in ["nb_new_designs"]:
        if key in kwargs:
            cmd += ["--" + key, str(kwargs[key])]
            cmd += ["--only_gener"]

    # Optional parameters without arguments (flags)
    for key in [
        "synth_only",
        "cmplx_only",
        "skip_synth",
        "skip_swact",
        "skip_power",
        "skip_cmplx",
    ]:
        if kwargs[key]:
            cmd += ["--" + key]

    # Advanced configurations
    if "bulk_flow_dirname" in kwargs and kwargs["bulk_flow_dirname"] is not None:
        cmd += ["--bulk_flow_dirname", kwargs["bulk_flow_dirname"]]
        if kwargs["bulk_flow_dirname"] == "power_out":
            cmd += ["--skip_synth", "--skip_swact"]
        if kwargs["bulk_flow_dirname"] == "synth_out":
            # We expect a flowy run here
            cmd += ["--skip_swact", "--skip_power"]
        else:
            raise NotImplementedError(
                f"bulk_flow_dirname {kwargs['bulk_flow_dirname']} is not implemented for the launher"
            )

    # Enforce some arguments
    cmd += ["--ignore_user_prompts"]
    cmd += ["--keep_not_valid"]

    # Discriminate task
    if "nb_new_designs" in kwargs:
        task = "generate"
    else:
        task = "launch"

    if task == "launch" and global_vars["is_slurm"]:
        # Split the task into subtasks
        # cmd = " ".join(cmd)

        dir_config = ConfigDir(
            output_dir_name=kwargs["output_dir_name"],
            experiment_name=kwargs["experiment_name"],
            bulk_flow_dirname=kwargs.get("bulk_flow_dirname", None),
        )
        list_of_valid_synth_design_numbers = get_list_of_synth_designs_number(dir_config)
        list_of_valid_design_numbers = list(
            get_list_of_gener_designs_number(
                dir_config, filter_design_numbers=list_of_valid_synth_design_numbers, filter_mode="exclude"
            )
        )

        cmd = " ".join(cmd)
        if kwargs.get("bulk_flow_dirname", None) is not None and kwargs.get("bulk_flow_dirname") == "synth_out":
            # We are running a flow run, so we want to have only a few designs per launcher
            nb_of_designs_per_launcher = int(24 / 12)
            all_list_of_valid_design_numbers = [
                list_of_valid_design_numbers[i : i + nb_of_designs_per_launcher]
                for i in range(0, len(list_of_valid_design_numbers), nb_of_designs_per_launcher)
            ]

            _cmd = []
            for sublist in all_list_of_valid_design_numbers:
                _cmd.append(cmd + " --design_number_list " + " ".join(sublist))
            cmd = _cmd

        else:
            pass

    else:
        # Run the task
        cmd = " ".join(cmd)

    return run_cmd(cmd, task=task, nb_workers=kwargs["nb_workers"], is_dry_run=kwargs["is_dry_run"])


def analyze_synthesis(**kwargs) -> CompletedProcess:
    """
    Build the command to run a analyze task and run it.
    """
    cmd = ["python", "src/genial/experiment/task_analyzer.py"]

    # Required parameters with arguments
    for key in [
        "experiment_name",
        "output_dir_name",
        "nb_workers",
    ]:
        cmd += ["--" + key, str(kwargs[key])]

    # Optional parameters with arguments
    for key in [
        "bulk_flow_dirname",
        "technology",
    ]:
        if key in kwargs and kwargs[key] is not None:
            cmd += ["--" + key, str(kwargs[key])]

    # Optional parameters without arguments (flags)
    for key in [
        "synth_only",
        "cmplx_only",
        "skip_synth",
        "skip_swact",
        "skip_power",
        "skip_cmplx",
        "rebuild_db",
    ]:
        if kwargs[key]:
            cmd += ["--" + key]

    # Enforce some arguments
    cmd += ["--fast_plots"]
    cmd += ["--ignore_user_prompts"]
    cmd += ["--keep_not_valid"]

    cmd = " ".join(cmd)
    return run_cmd(cmd, task="analyze", nb_workers=kwargs["nb_workers"], is_dry_run=kwargs["is_dry_run"])


def train_enc_to_score(**kwargs) -> CompletedProcess:
    """
    Build the command for running a training task and run it.
    """

    cmd = [
        "python",
        "src/genial/training/mains/trainer_enc_to_score_value.py",
    ]

    # Required parameters with arguments
    for key in [
        "experiment_name",
        "output_dir_name",
        "batch_size",
        "nb_workers",
        "max_epochs",
        "checkpoint_naming_style",
        "score_type",
        "score_rescale_mode",
        "check_val_every_n_epoch",
        "device",
        "yml_config_path",
        "trainer_version_number",
    ]:
        cmd += ["--" + key, str(kwargs[key])]

    # Optional parameters with arguments
    for key in [
        "bulk_flow_dirname",
    ]:
        if key in kwargs and kwargs[key] is not None:
            cmd += ["--" + key, str(kwargs[key])]

    # Optional parameters without arguments (flags)
    for key in [
        "synth_only",
        "cmplx_only",
        "skip_synth",
        "skip_swact",
        "skip_power",
        "skip_cmplx",
    ]:
        if kwargs[key]:
            cmd += ["--" + key]

    # Advanced parameters
    if kwargs["use_ssl_model"]:
        ssl_model_checkpoint = kwargs["ssl_model_checkpoint"]
        if ssl_model_checkpoint is None:
            raise ValueError("ssl_model_checkpoint must be specified if use_ssl_model is True")

        cmd += ["--model_checkpoint_path", ssl_model_checkpoint, "--trainer_task", "finetune_from_ssl"]

    # Enforce some command line arguments
    cmd += ["--ignore_user_prompts"]
    cmd += ["--keep_not_valid"]

    cmd = " ".join(cmd)
    return run_cmd(cmd, task="train", nb_workers=kwargs["nb_workers"], is_dry_run=kwargs["is_dry_run"])


def generate_prototypes(
    **kwargs,
) -> CompletedProcess:
    """
    Build the command for running a prototype generation task and run it.
    """

    cmd = ["python", "src/genial/utils/prototype_generator_v2.py"]

    # Required parameters with arguments
    for key in [
        "experiment_name",
        "output_dir_name",
        "dst_output_dir_name",
        "trainer_version_number",
        "nb_gener_prototypes",
        "batch_size_proto",
        "device",
        "yml_config_path",
        "max_epochs",
        "score_type",
        "nb_workers",
    ]:
        if key == "batch_size_proto":
            cmd += ["--batch_size", str(kwargs[key])]
        else:
            cmd += ["--" + key, str(kwargs[key])]

    # Optional parameters with arguments
    for key in [
        "bulk_flow_dirname",
    ]:
        if key in kwargs and kwargs[key] is not None:
            cmd += ["--" + key, str(kwargs[key])]

    # Optional parameters without arguments (flags)
    for key in [
        "do_prototype_pattern_gen",
        "synth_only",
        "cmplx_only",
        "skip_synth",
        "skip_swact",
        "skip_power",
        "skip_cmplx",
    ]:
        if kwargs[key]:
            cmd += ["--" + key]

    # Enforce some command line arguments
    cmd += ["--ignore_user_prompts"]
    cmd += ["--keep_not_valid"]

    cmd = " ".join(cmd)
    return run_cmd(cmd, task="recommend", nb_workers=kwargs["nb_workers"], is_dry_run=kwargs["is_dry_run"])


def merge_output_dirs(
    **kwargs,
) -> CompletedProcess:
    """
    Build the command for merging two outptu dirs together and run it.
    """

    cmd = [
        "python",
        "src/genial/utils/merge_output_dirs.py",
    ]

    # Required parameters with arguments
    for key in [
        "experiment_name",
        "output_dir_name_0",
        "output_dir_name_1",
        "dst_output_dir_name",
        "frac_0",
        "frac_1",
    ]:
        cmd += ["--" + key, str(kwargs[key])]

    # Optional parameters with arguments
    for key in [
        "bulk_flow_dirname",
    ]:
        if key in kwargs and kwargs[key] is not None:
            cmd += ["--" + key, str(kwargs[key])]

    # Optional parameters without arguments (flags)
    for key in [
        "synth_only",
        "cmplx_only",
        "skip_synth",
        "skip_swact",
        "skip_power",
        "skip_cmplx",
    ]:
        if kwargs[key]:
            cmd += ["--" + key]

    # Enforce some command line arguments
    cmd += ["--ignore_user_prompts"]
    cmd += ["--keep_not_valid"]
    cmd += ["--force"]

    cmd = " ".join(cmd)
    return run_cmd(cmd, task="merge", nb_workers=kwargs["nb_workers"], is_dry_run=kwargs["is_dry_run"])


def delete_directory(todelete_dir_name: str, **kwargs):
    """
    Delete the directory todelete_dir_name.
    """

    logger.info("")
    logger.info(f"|| --------------------------------------------------------------------------------------")
    logger.info(f"Deleting directory {todelete_dir_name}...")
    todelete_kwargs = {
        "experiment_name": kwargs["experiment_name"],
        "output_dir_name": todelete_dir_name,
        "trainer_version_number": kwargs["trainer_version_number"],
    }

    todelete_dir_config = ConfigDir(**todelete_kwargs)

    # Save model checkpoints before deleting
    save_ckpts_dirpath = (
        todelete_dir_config.root_output_dir.parent / kwargs["save_ckpt_dir_name"] / f"iter{kwargs['iteration']}"
    )
    if not save_ckpts_dirpath.exists():
        save_ckpts_dirpath.mkdir(exist_ok=True, parents=True)
    if not kwargs["is_dry_run"]:
        if todelete_dir_config.trainer_out_dir_ver.exists():
            shutil.copytree(todelete_dir_config.trainer_out_dir_ver, save_ckpts_dirpath, dirs_exist_ok=True)
    else:
        logger.info(
            f"Saving model checkpoints from {todelete_dir_config.trainer_out_dir_ver} to {save_ckpts_dirpath} skipped for dryrun"
        )

    # Delete the previous merge directory
    if todelete_dir_config.root_output_dir.exists():
        logger.info(f"Deleting {todelete_dir_config.root_output_dir}...")
        if not kwargs["is_dry_run"]:
            shutil.rmtree(todelete_dir_config.root_output_dir)
        else:
            logger.info(f"Deletion skipped for dry run.")

    else:
        logger.info("No previous merge to delete")
    logger.info(f"|| --------------------------------------------------------------------------------------")


# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
@dataclass
class ModelConfig:
    n_cls_token: int = 1
    max_restart_lr: float = 0.00025
    max_scratch_lr: float = 0.0005
    d_model: int = 512
    dim_feedforward: int = 1024
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    batch_size: int = 5000


def gen_temp_yaml_file(model_config: ModelConfig) -> str:
    # Generate a temporary YAML file with the model configuration with datestimestamp
    logger.info(asdict(model_config))
    temp_yaml_path = "temp_model_config" + str(int(time())) + ".yml"
    with open(temp_yaml_path, "w") as f:
        yaml.dump(model_config.__dict__, f)
    return temp_yaml_path


def get_dir_name(prefix: str, desc: str, generation: int) -> str:
    """
    Generate a directory name with the given prefix, description, and generation number.
    """
    assert prefix is not None
    assert generation is not None
    return f"{prefix}_{desc}_iter{generation}"


# --------------------------------------------------------
# Initial Generation
# --------------------------------------------------------
def initial_generation(**kwargs) -> str:
    # 1.a) Launcher (Generate)
    if not kwargs["skip_init_gener"]:
        launch_task(
            **kwargs,
            nb_new_designs=kwargs["nb_init_designs"],
            output_dir_name=kwargs["initial_dataset"],
        )

    # 1.b) Launcher (Synthesize)
    if not kwargs["skip_restart_launch"]:
        launch_task(
            **kwargs,
            output_dir_name=kwargs["initial_dataset"],
        )

    # 2. Synthesis Analysis
    analyze_synthesis(
        **kwargs,
        output_dir_name=kwargs["initial_dataset"],
        rebuild_db=True,
    )

    return kwargs["initial_dataset"]


# --------------------------------------------------------
# Loop
# --------------------------------------------------------
def run_loop(**kwargs) -> str:
    output_dir_name = kwargs["dir_name"]
    iteration = kwargs.pop("iteration")
    logger.info(iteration)
    _gen_dir_name = kwargs.get("gen_dir_name", get_dir_name(output_dir_name, "gen", iteration))
    proto_dir_name = get_dir_name(output_dir_name, "proto", iteration)
    merge_dir_name = get_dir_name(output_dir_name, "merge", iteration)

    skip_proto = False
    skip_merge = False
    # restart_step comes from detection in main():
    # - None: normal flow (train -> proto -> launch -> analyze -> merge)
    # - "proto": we already have prototypes for this iteration; skip training + proto generation,
    #             then continue with launch + analyze + merge.
    # - "merge": full iteration is already merged; skip all heavy work and just re-analyze merge dir
    #             (useful for quick resume and to keep downstream plots in sync), then move to next iter.
    restart_step = kwargs.pop("restart_step")
    if restart_step is not None:
        if restart_step == "merge":
            skip_merge = True
        if restart_step == "proto":
            skip_proto = True

    _trainer_version_number = 0
    skip_launch = kwargs.get("skip_launch", False)
    # If True, skip training but still run prototype generation.
    skip_training = kwargs.get("skip_training", False)

    # When skip_merge is True, we already have a complete merge dir for this iteration.
    # We therefore only re-run the analysis on the existing merge dir (see below) and return it.
    if not skip_merge:
        if not skip_proto:
            if not skip_training:
                train_enc_to_score(
                    output_dir_name=_gen_dir_name,
                    batch_size=kwargs["batch_size_trainer"],
                    max_epochs=kwargs["max_epochs_trainer"],
                    checkpoint_naming_style="enforce_increase",
                    score_rescale_mode="standardize",
                    check_val_every_n_epoch=1,
                    **kwargs,
                )

            generate_prototypes(
                do_prototype_pattern_gen=True,
                output_dir_name=_gen_dir_name,
                dst_output_dir_name=proto_dir_name,
                batch_size=kwargs["batch_size_proto"],
                max_epochs=kwargs["max_epochs_proto"],
                **kwargs,
            )

        # If resuming from "proto" and --skip_restart_launch is provided, we skip relaunching flows
        # for existing prototypes; analysis below will still refresh DB consistency.
        if not skip_launch:
            launch_task(
                output_dir_name=proto_dir_name,
                **kwargs,
            )

        analyze_synthesis(
            output_dir_name=proto_dir_name,
            rebuild_db=True,
            **kwargs,
        )

        merge_output_dirs(
            output_dir_name_0=_gen_dir_name,
            output_dir_name_1=proto_dir_name,
            dst_output_dir_name=merge_dir_name,
            frac_0=1,
            frac_1=1,
            **kwargs,
        )

    analyze_synthesis(
        output_dir_name=merge_dir_name,
        rebuild_db=True,
        **kwargs,
    )

    # Delete the directory containing the previous merge to save space
    if kwargs["delete_merge_dirs"] and iteration >= 1:
        prev_merge_dir_name = get_dir_name(output_dir_name, "merge", iteration - 1)
        delete_directory(todelete_dir_name=prev_merge_dir_name, iteration=iteration - 1, **kwargs)

    return merge_dir_name


def main():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--is_slurm", action="store_true")
    arg_parser.add_argument("--do_ssl_only", action="store_true")
    arg_parser.add_argument("--is_dry_run", action="store_true")
    # Deletion policy for previous merge directories
    # Default is now to delete; use --KEEP_merge_dirs (or --keep_merge_dirs / --skip_merge_dirs) to preserve.
    arg_parser.add_argument("--delete_merge_dirs", action="store_true", help="[DEPRECATED] Deletion is now default.")
    arg_parser.add_argument(
        "--KEEP_merge_dirs",
        "--keep_merge_dirs",
        "--skip_merge_dirs",
        action="store_true",
        dest="keep_merge_dirs",
        help="Keep previous merge directories instead of deleting them (default is delete).",
    )
    arg_parser.add_argument("--skip_restart_launch", action="store_true")
    # Restart control: keep default behavior (auto-detect), but allow explicit override.
    arg_parser.add_argument(
        "--resume_from",
        type=str,
        choices=["auto", "proto", "merge", "none"],
        default="auto",
        help=(
            "Resume mode. 'auto' infers last completed step; 'proto' resumes at launch/merge for the latest proto_*, "
            "'merge' resumes after the latest merge_* (reanalyzes then continues); 'none' starts fresh ignoring existing outputs."
        ),
    )
    arg_parser.add_argument(
        "--config_filepath",
        type=str,
        default=os.environ["SRC_DIR"] + "/scripts/loop_scripts/configs/example_default_loop.yaml",
    )
    arg_parser.add_argument(
        "--skip_init_gener",
        action="store_true",
        help="When set, the initial design generation will be skipped, but the initial launch will be run.",
    )
    arg_parser.add_argument(
        "--do_init_analysis",
        action="store_true",
        help="When set, the initial dataset analysis will be done. If the user doesn't want to run the inital launch, the user should then make sure the `--skip_restart_launch` is set.",
    )
    arg_parser.add_argument(
        "--do_init_gener_n_launch",
        action="store_true",
        help="When set, the initial generation and launch will be realized, even if an initial dir name is given. Generation will be skipped if --skip_init_gener is set.",
    )

    args = arg_parser.parse_known_args()[0]

    config_filepath = Path(args.config_filepath)

    # Define parameters
    # To remove a parameter, if it's not a boolean, set it to None.
    if config_filepath is None or not config_filepath.exists():
        err_msg = f"Provided configuration file path does not exist. {config_filepath}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    else:
        logger.warning(f"Loading run configuration from file")
        logger.warning(config_filepath)
        run_config = yaml.safe_load(config_filepath.read_text())
    run_config.update(vars(args))
    run_config["save_ckpt_dir_name"] = run_config["dir_name"] + "_saved_ckpts"
    # Apply new default for merge dir deletion: delete by default, unless keep was requested
    if run_config.get("keep_merge_dirs", False):
        run_config["delete_merge_dirs"] = False
    elif "delete_merge_dirs" in run_config:
        # If user explicitly provided this (via YAML or CLI), keep it, but warn if True since it's default
        if run_config.get("delete_merge_dirs"):
            logger.warning(
                "--delete_merge_dirs is deprecated; deletion is the default. Use --KEEP_merge_dirs to preserve instead."
            )
    else:
        # Not specified anywhere -> default to delete
        run_config["delete_merge_dirs"] = True

    skip_restart_launch = None
    do_init_launch = False
    if run_config["initial_dataset"] is None:
        do_init_launch = True
        run_config["initial_dataset"] = get_dir_name(
            prefix=run_config["dir_name"],
            desc="gen",
            generation=0,
        )

        # Check if the initial dataset exists
        initial_dataset_dir_path = (
            Path(os.environ.get("WORK_DIR")) / "output" / run_config["experiment_name"] / run_config["initial_dataset"]
        )
        print(initial_dataset_dir_path)
        if initial_dataset_dir_path.exists():
            run_config["skip_init_gener"] = True
            logger.warning(
                f"Initial dataset {run_config['initial_dataset']} already exists. Skipping initial generation."
            )
        else:
            run_config["skip_init_gener"] = False
        # If the initial dataset is not given, we assume we want to do the initial generation and launch

        # Check if there is already an iteration 0 dir
        proto_dir_name_0 = get_dir_name(run_config["dir_name"], "proto", 0)
        proto_dir_path = Path(os.environ.get("WORK_DIR")) / "output" / run_config["experiment_name"] / proto_dir_name_0
        if proto_dir_path.exists():
            skip_restart_launch = True
            do_init_launch = False
        else:
            skip_restart_launch = False
            do_init_launch = True

    elif run_config["do_init_gener_n_launch"]:
        do_init_launch = True
    elif run_config["do_init_analysis"]:
        do_init_launch = True
        skip_restart_launch = args.skip_restart_launch

    # Setup logging
    init_dir_kwargs = {
        "experiment_name": run_config["experiment_name"],
        "output_dir_name": run_config["initial_dataset"],
        "trainer_version_number": 0,
        "bulk_flow_dirname": run_config["bulk_flow_dirname"],
    }
    initi_dir_config = ConfigDir(**init_dir_kwargs)
    log_filepath = initi_dir_config.root_output_dir / "full_run_log.log"
    if Path(log_filepath).exists():
        log_filepath.unlink()
    # Reset sinks: remove default console handler to control what reaches terminal
    logger.remove()
    # File sink: capture everything to the run log
    logger.add(log_filepath, level="INFO")
    # Console sink: only show logs emitted from this script
    logger.add(sys.stderr, level="INFO", colorize=True, filter=lambda r: r.get("module") == "full_run_v2")
    logger.info(f"Log file has been set to:")
    logger.info(f"{log_filepath}")

    logger.info(f"Starting run with configuration:")
    logger.info(f"{run_config}")

    # Save config in root output dir name as a yaml file
    full_run_config_filepath = initi_dir_config.root_output_dir / "full_run_config.yaml"
    with open(full_run_config_filepath, "w") as f:
        yaml.dump(run_config, f)
    logger.info(f"Full run configuration has been saved to:")
    logger.info(f"{full_run_config_filepath}")

    # Clarify resume from initial dataset:
    # If generation exists for the initial dataset but synthesis is incomplete,
    # force running the initial launch (synthesis) regardless of how the script was interrupted.
    # This handles the case where the initial launch was interrupted but designs were already generated.
    if run_config.get("resume_from", "auto") != "none":
        try:
            gen_nums = list(get_list_of_gener_designs_number(initi_dir_config))
            synth_nums = list(get_list_of_synth_designs_number(initi_dir_config))
            logger.info(f"Initial dataset status: generated={len(gen_nums)} synthesized={len(synth_nums)}")
            if len(gen_nums) > 0 and len(synth_nums) < len(gen_nums):
                # We have generated designs but synthesis is not complete -> ensure we run initial launch
                do_init_launch = True
                skip_restart_launch = False
                logger.info(
                    "Detected partial initial synthesis; will resume initial launch on existing generated designs."
                )
            elif len(gen_nums) > 0 and len(synth_nums) >= len(gen_nums):
                # Everything synthesized -> skip initial launch by default unless explicitly requested
                if skip_restart_launch is None:
                    skip_restart_launch = True
                logger.info("Initial dataset already fully synthesized; skipping initial launch.")
        except Exception as e:
            logger.warning(f"Could not auto-detect initial dataset state ({e}); keeping CLI defaults.")

    # Find initial generation / resume point
    # We scan for existing proto_*/merge_* directories to determine where to resume.
    proto_dir_name = get_dir_name(run_config["dir_name"], "proto", 0)
    merge_dir_name = get_dir_name(run_config["dir_name"], "merge", 0)
    proto_dir_name_temp = proto_dir_name.replace("iter0", "")
    merge_dir_name_temp = merge_dir_name.replace("iter0", "")
    exp_dirpath: Path = initi_dir_config.root_output_dir.parent
    max_iter_dict = {"proto": -1, "merge": -1}
    for dir_name in exp_dirpath.iterdir():
        if proto_dir_name_temp in dir_name.name:
            iter_nb = int(dir_name.name.split("iter")[1])
            max_iter_dict["proto"] = max(max_iter_dict["proto"], iter_nb)
        if merge_dir_name_temp in dir_name.name:
            iter_nb = int(dir_name.name.split("iter")[1])
            max_iter_dict["merge"] = max(max_iter_dict["merge"], iter_nb)

    restart_iteration = 0
    restart_step = None
    gen_dir_name = run_config["initial_dataset"]
    # Respect explicit resume override if provided; otherwise keep auto-detection behavior.
    resume_from = run_config.get("resume_from", "auto")
    logger.info(f"Resume mode: {resume_from}")
    if resume_from == "none":
        # Explicitly disable resume
        logger.info("Auto-resume disabled (resume_from=none). Starting fresh.")
    else:
        # Compute an auto-resume suggestion first
        auto_restart_iteration = 0
        auto_restart_step = None
        auto_gen_dir_name = gen_dir_name
        if max_iter_dict["proto"] != -1:
            if max_iter_dict["merge"] == max_iter_dict["proto"]:
                # Latest proto and merge are for the same iter -> iteration fully merged
                auto_restart_iteration = max_iter_dict["merge"]
                auto_restart_step = "merge"
                auto_gen_dir_name = get_dir_name(run_config["dir_name"], "merge", auto_restart_iteration)
            else:
                # Latest proto is newer than latest merge -> resume from proto stage
                auto_restart_iteration = max_iter_dict["proto"]
                auto_restart_step = "proto"
                if auto_restart_iteration == 0:
                    auto_gen_dir_name = get_dir_name(run_config["dir_name"], "gen", auto_restart_iteration)
                else:
                    auto_gen_dir_name = get_dir_name(run_config["dir_name"], "merge", auto_restart_iteration - 1)

        # Apply override if requested
        if resume_from == "auto":
            restart_iteration = auto_restart_iteration
            restart_step = auto_restart_step
            gen_dir_name = auto_gen_dir_name
        elif resume_from == "proto":
            if max_iter_dict["proto"] != -1:
                restart_iteration = max_iter_dict["proto"]
                restart_step = "proto"
                if restart_iteration == 0:
                    gen_dir_name = get_dir_name(run_config["dir_name"], "gen", restart_iteration)
                else:
                    gen_dir_name = get_dir_name(run_config["dir_name"], "merge", restart_iteration - 1)
            else:
                logger.warning("resume_from=proto requested but no proto_* directories found; starting fresh.")
        elif resume_from == "merge":
            if max_iter_dict["merge"] != -1:
                restart_iteration = max_iter_dict["merge"]
                restart_step = "merge"
                gen_dir_name = get_dir_name(run_config["dir_name"], "merge", restart_iteration)
            else:
                logger.warning("resume_from=merge requested but no merge_* directories found; starting fresh.")

        # If we intend to resume from a proto step, verify the token exists; otherwise re-run ONLY prototype generation
        skip_training_on_resume = False
        if restart_step == "proto":
            intended_proto_dir = exp_dirpath / get_dir_name(run_config["dir_name"], "proto", restart_iteration)
            token_path = intended_proto_dir / "proto_generation_done.token"
            if not token_path.exists():
                logger.warning(
                    f"Proto resume requested at {intended_proto_dir}, but token not found; will re-run prototype generation (skip training)."
                )
                # Run proto generation again, but do not re-train
                skip_training_on_resume = True
                restart_step = None

        if restart_step is not None:
            logger.info(
                f"Resuming at iter={restart_iteration} after step={restart_step}; base dataset set to: {gen_dir_name}"
            )

    # Check whether lauch task should be operated on restart
    if skip_restart_launch is None:
        if restart_step is not None:
            skip_restart_launch = args.skip_restart_launch
        else:
            skip_restart_launch = False

    # Check slurm setup
    if args.is_slurm:
        SlurmDispatcher.check_slurm_setup(**run_config)

    try:
        # Initial generation
        if do_init_launch:
            logger.info(f"|| ======================================================================================")
            logger.info(f"|| Starting initial generation ...")
            start = time()
            gen_dir_name = initial_generation(
                iteration=0,
                **run_config,
            )
            end = time()
            logger.info(
                f"Completed initial generation | Generated directory: {run_config['experiment_name']}{gen_dir_name} | Time: {end - start:.2f} seconds"
            )
            logger.info(f"|| ======================================================================================")
        else:
            logger.info("")
            logger.info("")
            logger.info("")
            logger.info(f"|| ======================================================================================")
            logger.info(f"|| Initial generation skipped. ")
            # gen_dir_name = run_config["initial_dataset"]
            logger.info(f"|| ======================================================================================")

        # Ensure local flag exists even if auto-resume chose 'none'
        skip_training_on_resume = locals().get("skip_training_on_resume", False)

        # Run the loop
        for i in range(restart_iteration, run_config["n_iter"]):
            logger.info("")
            logger.info("")
            logger.info("")
            logger.info(f"|| ======================================================================================")
            logger.info(f"|| Starting iteration {i}...")
            start = time()
            gen_dir_name = run_loop(
                gen_dir_name=gen_dir_name,
                iteration=i,
                restart_step=restart_step,
                skip_launch=skip_restart_launch,
                skip_training=skip_training_on_resume,
                **run_config,
            )
            skip_restart_launch = False
            restart_step = None
            skip_training_on_resume = False
            end = time()
            logger.info(
                f"Completed iteration {i} | Generated directory: {run_config['experiment_name']}{gen_dir_name} | Time: {end - start:.2f} seconds"
            )
            logger.info(f"|| ======================================================================================")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
        logger.error(f"Now Exiting ...")

    logger.info(f"Full log can be found in: {log_filepath}")


if __name__ == "__main__":
    main()
