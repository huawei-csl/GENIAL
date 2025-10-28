from typing import Any
import os
from pathlib import Path
import shutil

from loguru import logger
import git
from time import gmtime, strftime

import genial.experiment.binary_operators as bin_ops

import json

from typing import Literal

from genial.config.arg_parsers import get_default_parser


class ConfigDir:
    __valid_encoding_types__ = ["unsigned_int", "twos_comp", "one_hot_msbl", "one_hot_msbr", "twos_comp_both", "mixed"]
    __valid_design_types__ = ["adder", "multiplier", "encoder", "decoder", "fsm"]

    __valid_generation_options__ = {
        "permute_in_mode": ["col_permutation_from_standard", "classic_encoding"],
        "permute_out_mode": ["keep_in_repr", "random"],
    }

    def __init__(self, is_analysis: bool = False, is_merging: bool = False, **args_dict) -> None:
        logger.info(f"Setting up ConfigDir ...")
        if len(args_dict) == 0:
            args_dict = get_default_parser()
        self.args_dict = args_dict

        # Get synth and swact versions
        self.synth_ver = self.args_dict.get("synth_version", 0)
        self.swact_ver = self.args_dict.get("swact_version", 0)
        self.power_ver = self.args_dict.get("power_version", 0)

        # Specify Output Dir Name
        # if self.args_dict.get("debug", False):
        # _output_dir_name = "debug"
        if self.args_dict.get("output_dir_name", None) is not None:
            _output_dir_name = self.args_dict.get("output_dir_name", None)
        else:
            if is_analysis:
                logger.error(
                    f"Running analysis without specifying the output_dir_name, are you sure your command to launch this script is correct?"
                )
                answer = input("Answer [Y]es or [N]o.")
                if answer.lower().startswith("y"):
                    _output_dir_name = ""
                    pass
                else:
                    exit(1)
            else:
                _output_dir_name = strftime("%Y-%m-%d_%H-%M", gmtime())
                repo = git.Repo(search_parent_directories=True)
                _output_dir_name += "_" + str(repo.git.rev_parse(repo.head, short=True))
                _output_dir_name += f"_SYNTHV{self.synth_ver}"

        if _output_dir_name is not None or _output_dir_name != "debug":
            if "SYNTHV" in _output_dir_name:
                self.synth_ver = int(_output_dir_name.split("SYNTHV")[1].split("_")[0])
            if "SWACTV" in _output_dir_name:
                self.swact_ver = int(_output_dir_name.split("SWACTV")[1].split("_")[0])

        self.output_dir_name = _output_dir_name
        self.args_dict["output_dir_name"] = _output_dir_name
        self.experiment_name = self.args_dict.get("experiment_name", None)
        if self.experiment_name is None:
            error_msg = (
                f"Please provide a valid experiment name."
                f"It must match with one of the experiment folder name in {os.getenv('SRC_DIR')}/src/genial/templates_and_launch_scripts"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if os.environ.get("WORK_DIR") is None:
            logger.error(f"Source directory `SRC_DIR` set in `.env` file does not exist.")
            logger.error(f"Did you source your environment before launching?")
            raise FileExistsError()
        else:
            # Where to store the output data
            self.work_dir = Path(os.environ.get("WORK_DIR"))

        if os.environ.get("SRC_DIR") is None:
            logger.error(f"Target directory `WORK_DIR` set in `.env` file does not exist.")
            logger.error(f"Did you source your environment before launching?")
            raise FileExistsError()
        else:
            # Where are the files of this repository
            self.src_dir = Path(os.environ.get("SRC_DIR"))

        # Input directories
        self.exp_templates_dir = self.src_dir / "src/genial/templates_and_launch_scripts" / self.experiment_name
        self.default_templates_dir = self.src_dir / "src/genial/templates_and_launch_scripts" / "0_defaults"

        # Output directories
        self.root_output_dir: Path = self.work_dir / "output" / self.experiment_name / self.output_dir_name
        self.root_output_dir.mkdir(parents=True, exist_ok=True)

        logger.opt(colors=True).info(f"<red>Root output directory</red> set to: {self.root_output_dir}")

        if is_analysis and not self.root_output_dir.exists():
            logger.error(f"Specified root output directory does not exist.")
            logger.error(self.root_output_dir)
            raise FileExistsError(f"Specified root output directory does not exist.")

        elif is_merging and self.root_output_dir.exists():
            if args_dict.get("force", False):
                if not args_dict.get("ignore_user_prompts"):
                    logger.warning(f"`force` is set, the output dir will be entirely erased. Continue?")
                    do_it = str(input("y/n")).lower()[0]
                else:
                    logger.warning(f"Merge directory already exists but `force` is set, pursuing.")
                    do_it = "y"
                if do_it == "y":
                    shutil.rmtree(self.root_output_dir)
            else:
                logger.error(
                    "Specified root output directory exists while merging has been requested. Please ensure that the "
                    "destination output directory does not exists when calling the merging script."
                )
                logger.error(self.root_output_dir)
                exit(1)

        self.root_output_dir.mkdir(parents=True, exist_ok=True)

        self.generation_out_dir = self.root_output_dir / "generation_out"
        self.generation_out_dir.mkdir(exist_ok=True)

        self.synth_out_dir = self.root_output_dir / "synth_out"
        self.synth_out_dir.mkdir(exist_ok=True)

        self.swact_out_dir = self.root_output_dir / "test_out"
        self.swact_out_dir.mkdir(exist_ok=True)

        self.power_out_dir = self.root_output_dir / "power_out"
        self.power_out_dir.mkdir(exist_ok=True)

        self.analysis_out_dir = self.root_output_dir / "analysis_out"
        self.analysis_out_dir.mkdir(exist_ok=True)

        self.trainer_out_root_dir = self.root_output_dir / "trainer_out"
        self.trainer_out_root_dir.mkdir(exist_ok=True)
        self.trainer_pl_logs_dir = self.trainer_out_root_dir / "lightning_logs"  # Will be made by pytorch lightning

        self.recommender_out_root_dir = self.root_output_dir / "recommender_out"
        self.recommender_out_root_dir.mkdir(exist_ok=True)

        self.bulk_flow_dirname = self.args_dict.get("bulk_flow_dirname", None)
        if self.args_dict.get("bulk_flow_dirname", None) is not None:
            self._switch_to_bulk_flow_mode()
            self.is_bulk_flow_mode = True
        else:
            self.is_bulk_flow_mode = False

        self.technology = self.args_dict.get("technology", "notech_yosys")
        self.cell_cost_model = self.args_dict.get("cell_cost_model", "transistor")

        self.setup_trainer_version_output_dirs()

        self.exp_config = None
        self.exp_config = self.get_experiment_configuration()

        _encoder_out_dir = self.exp_config.get("encoder_generation_out_dir", None)
        if _encoder_out_dir is None or _encoder_out_dir == "":
            logger.warning(
                f"Path to associated encoder design is not specified. To configure, please add `encoder_generation_out_dir` argument to experiment configuration file {self.exp_templates_dir / 'experiment_configuration.json'}."
            )
            self.encoder_out_dir = None
        else:
            self.encoder_out_dir = self.work_dir / "output" / _encoder_out_dir
            if not self.encoder_out_dir.exists():
                logger.warning(
                    f"Provided path to associated encoder design does not exist {self.encoder_out_dir}. Please check it in {self.exp_templates_dir / 'experiment_configuration.json'}."
                )
                self.encoder_out_dir = None
            logger.info(
                f"Path to encoder directory has been configured as {self.encoder_out_dir}. Encoder correlation will be realized during SwAct analysis."
            )

        self.special_designs_filepath = self.root_output_dir / "special_designs.json"

        # logger.opt(colors=True).info(f"<red>Output Directories</red> initialized in {self.root_output_dir}.")
        logger.info(f"ConfigDir initialized.\n")

    def setup_trainer_version_output_dirs(self, trainer_version_number: int | None = None) -> bool:
        """
        Tries to setup the versioned output directories for the trainer and recommenders.
        Args:
            trainer_version_number: int|None, has precendence over self.args_dict.get("trainer_version_number")
        Return:
            False if successful
            True if fails
        """

        if trainer_version_number is not None:
            _trainer_version_number = trainer_version_number
            self.args_dict["trainer_version_number"] = _trainer_version_number
            logger.warning(f"Trainer version number has been set to {_trainer_version_number} in ConfigDir.args_dict.")
        elif self.args_dict.get("trainer_version_number") is not None:
            _trainer_version_number = self.args_dict.get("trainer_version_number", None)
        else:
            _trainer_version_number = None

        if _trainer_version_number is not None:
            self.recommender_out_dir_ver = self.recommender_out_root_dir / f"version_{_trainer_version_number}"
            self.recommender_out_dir_ver.mkdir(exist_ok=True)

            self.trainer_out_dir_ver = self.trainer_pl_logs_dir / f"version_{_trainer_version_number}"
            self.recommender_out_dir_ver.mkdir(exist_ok=True)
            return False
        else:
            self.recommender_out_dir_ver = None
            self.trainer_out_dir_ver = None
            return True

    def get_experiment_configuration(self, args_dict: dict[str, Any] = None, force: bool = False) -> dict[str, str]:
        """Returns the design type based on the generator configuration file located in experiment template folder."""

        if self.exp_config is None or force:
            # Read json file
            design_config_filepath = self.exp_templates_dir / "experiment_configuration.json"
            exp_config_dict = json.loads(design_config_filepath.read_bytes())

            # Do some checks
            if exp_config_dict["design_type"] not in self.__valid_design_types__:
                raise ValueError(
                    f"Experiment name should be defined as `<design_type>_[...]` with design type among "
                    f"{self.__valid_design_types__}"
                )

            # Retro-compatibility
            if "input_encoding_type" not in exp_config_dict.keys():
                exp_config_dict["input_encoding_type"] = "twos_comp"
            else:
                if exp_config_dict["input_encoding_type"] not in self.__valid_encoding_types__:
                    raise ValueError(
                        f"Input encoding type {exp_config_dict['input_encoding_type']} not implemented. Should be "
                        f"in {self.__valid_encoding_types__}. Please modify the design_config.json file in the experiment template folder."
                    )
            if "output_encoding_type" not in exp_config_dict.keys():
                exp_config_dict["output_encoding_type"] = "twos_comp"
            else:
                if exp_config_dict["output_encoding_type"] not in self.__valid_encoding_types__:
                    raise ValueError(
                        f"Output encoding type {exp_config_dict['output_encoding_type']} not implemented. Should be in "
                        f"{self.__valid_encoding_types__}. Please modify the design_config.json file in the experiment template folder."
                    )

            # Get the list of inputs and outputs
            design_type = exp_config_dict["design_type"]
            io_ports_lists_filepath = self.default_templates_dir / "configuration" / f"{design_type}.json"
            io_ports_lists_filepath = json.loads((io_ports_lists_filepath).read_bytes())
            if "input_ports" not in io_ports_lists_filepath.keys():
                raise ValueError(f"Please the list of input port names in  {io_ports_lists_filepath}")
            if "output_ports" not in io_ports_lists_filepath.keys():
                raise ValueError(f"Please the list of output port names in  {io_ports_lists_filepath}")
            if not isinstance(io_ports_lists_filepath["input_ports"], list):
                raise ValueError(
                    f"The list of input port names `input_ports` should be given as a list of string in "
                    f"{io_ports_lists_filepath}"
                )
            if not isinstance(io_ports_lists_filepath["output_ports"], list):
                raise ValueError(
                    f"The list of output port names `output_ports` should be given as a list of string in "
                    f"{io_ports_lists_filepath}"
                )

            # Update the experiment configuration dictionary
            exp_config_dict.update(io_ports_lists_filepath)

            # Check whether synthesis must be run with flowy
            if "synth_with_flowy" not in exp_config_dict.keys():
                exp_config_dict["synth_with_flowy"] = False

            # Get some potential input command line argument to generate special configurations
            if args_dict is None:
                _args_dict = self.args_dict
            else:
                _args_dict = args_dict
            for key in ConfigDir.__valid_generation_options__.keys():
                if _args_dict.get(key, None) is not None:
                    msg = f"Generation option received: {key} = {_args_dict[key]}"
                    if _args_dict[key] in ConfigDir.__valid_generation_options__[key]:
                        exp_config_dict[key] = _args_dict[key]
                        logger.info(msg)
                    else:
                        logger.error(msg)
                        raise NotImplementedError(msg)

            return exp_config_dict
        else:
            return self.exp_config

    def _switch_to_bulk_flow_mode(self) -> None:
        """
        Switches the dir_config to the bulk flow mode where all files are outputed in the same folder.
        For instance, there is a power extraction flow where the full flow [synth + simulation (with SwAct acquisition!) + power extraction] is run in a single flow.
        """

        assert self.bulk_flow_dirname is not None, (
            "An unexected error occured. _switch_to_bulk_flow_mode() requires a bulk_flow_dirname argument and should not be called manually."
        )

        valid_bulk_flow_dirnames = ["synth_out", "test_out", "power_out", "generation_out"]
        if self.bulk_flow_dirname not in valid_bulk_flow_dirnames:
            logger.error(f"Argument `bulk_flow_dirname` should be among {valid_bulk_flow_dirnames}")
            raise ValueError(f"Argument `bulk_flow_dirname` should be among {valid_bulk_flow_dirnames}")

        self.synth_out_dir = self.root_output_dir / self.bulk_flow_dirname
        self.swact_out_dir = self.root_output_dir / self.bulk_flow_dirname
        self.power_out_dir = self.root_output_dir / self.bulk_flow_dirname
        logger.info("Argument `bulk_flow_dirname` received, config_dir has been set to bulk_flow_mode.")

        return None

    def update_experiment_configuration(self, args_dict: dict[str, Any]) -> dict[str, str]:
        self.exp_config = self.get_experiment_configuration(args_dict=args_dict, force=True)

    def get_bounding_values(
        self,
    ) -> dict[
        Literal[
            "min_val_out_included",
            "max_val_out_included",
            "min_val_in_included",
            "max_val_in_included",
        ],
        int,
    ]:
        """
        Returns a dictionnary containing the values for the max and min of inputs and outputs.
        """
        # Get limit values
        exp_config = self.get_experiment_configuration()
        in_enc_type = exp_config["input_encoding_type"]
        out_enc_type = exp_config["output_encoding_type"]
        in_bitwidth = int(exp_config["input_bitwidth"])
        out_bitwidth = int(exp_config["output_bitwidth"])
        if in_enc_type in ["twos_comp", "twos_comp_both", "mixed"]:
            min_val_in_included = bin_ops.min_value_tc(in_bitwidth)
            max_val_in_included = bin_ops.max_value_tc(in_bitwidth)
        elif in_enc_type == "unsigned_int":
            min_val_in_included = bin_ops.min_value_uint(in_bitwidth)
            max_val_in_included = bin_ops.max_value_uint(in_bitwidth)
        else:
            raise NotImplementedError()

        if out_enc_type == "twos_comp":
            min_val_out_included = bin_ops.min_value_tc(out_bitwidth)
            max_val_out_included = bin_ops.max_value_tc(out_bitwidth)
        elif out_enc_type == "twos_comp_both":
            min_val_out_included = bin_ops.min_value_tc(in_bitwidth + 2)
            max_val_out_included = bin_ops.max_value_tc(in_bitwidth + 2)
        elif out_enc_type == "unsigned_int":
            min_val_out_included = bin_ops.min_value_uint(out_bitwidth)
            max_val_out_included = bin_ops.max_value_uint(out_bitwidth)
        else:
            raise NotImplementedError()

        out_dict = {
            "min_val_out_included": min_val_out_included,
            "max_val_out_included": max_val_out_included,
            "min_val_in_included": min_val_in_included,
            "max_val_in_included": max_val_in_included,
        }

        return out_dict

    def find_all_temp_files(self, step: str = "generation") -> list[Path]:
        return self._get_shared_files(step=step, temp_only=True)

    def _get_shared_files(self, step: str = "synthesis", temp_only: bool = False) -> list[Path]:
        """
        Get all files that should be copied and used for the specified step
        These files are stored in templates files
        Experiment files have precedence over default files (based on file name)
        """
        assert step in ["analysis", "evaluation", "synthesis", "generation", "power_extraction"]

        # Get files version
        if step == "synthesis":
            dirname = step + f"_v{self.synth_ver}"
        elif step == "evaluation":
            dirname = step
            # Next version should use:
            # dirname = step + f"_v{self.swact_ver}"
        elif step == "power_extraction":
            dirname = step + f"_v{self.power_ver}"
        else:
            dirname = step
        default_templates_dir = self.default_templates_dir / dirname
        exp_templates_dir = self.exp_templates_dir / dirname

        shared_files = []
        shared_file_names = []

        if exp_templates_dir.exists():
            # Copy all experiment files
            for filepath in exp_templates_dir.iterdir():
                if filepath.is_file():
                    if temp_only and not filepath.name.endswith(".temp"):
                        pass
                    else:
                        shared_files.append(filepath)
                        shared_file_names.append(filepath.name)
                else:
                    logger.warning(
                        f"Element {filepath.name} in {exp_templates_dir} is not a file, make sure that the synthesis templates folder contains only files. {filepath.name} was not copied"
                    )

        # Copy all default files that do not already exist
        for filepath in default_templates_dir.iterdir():
            if filepath.is_file():
                # Get default file only if it is not overloaded by experiemnt specific file
                if filepath.name not in shared_file_names:
                    if temp_only and not filepath.name.endswith(".temp"):
                        pass
                    else:
                        shared_files.append(filepath)
                else:
                    logger.warning(f"Default file {filepath.name} has been overloaded by similar custom file.")
            else:
                logger.warning(
                    f"Element {filepath.name} in {default_templates_dir} is not a file, make sure that the synthesis templates folder contains only files. {filepath.name} was not copied"
                )

        return shared_files

    @staticmethod
    def read_special_designs(self) -> dict[str, list[str]] | None:
        """Checks the special_designs.json file in the root output directory."""
        # special_design_filepath = self.root_output_dir/"special_designs.json"
        if not self.special_designs_filepath.exists():
            logger.warning(f"Special design file does not exist. Running without description of special designs:")
            logger.info(self.special_designs_filepath)
            return {"legend": [], "design_numbers": []}
        else:
            return json.loads((self.special_designs_filepath).read_bytes())

    @staticmethod
    def setup_other_dir_config(args_dict) -> object:
        """
        Helper function to instantiate a `dir_config` of another `output_dir_name` from a path
        """

        if args_dict.get("other_output_dirpath", None) is not None:
            other_output_dirpath = Path(args_dict.get("other_output_dirpath"))

            # Do some checks
            assert other_output_dirpath.exists()
            if not other_output_dirpath.parent.name == args_dict.get("experiment_name"):
                logger.error(
                    f"Please make sure that the leaf names of `other_output_dirpath` provided matches with the pattern <experiement_name>/<any>"
                )
                logger.error(
                    f"i.e., make sure that the other synthesized dataset's output_dir sits next to the dataset's output_dir you want to synthesize/simulate"
                )
                raise ValueError(
                    f"Please make sure that the leaf names of `other_output_dirpath` provided matches with the pattern <experiement_name>/<any>"
                )

            # Setup the dir_config object
            other_synth_dir_config = ConfigDir(
                is_analysis=True,
                experiment_name=args_dict.get("experiment_name"),
                output_dir_name=other_output_dirpath.name,
            )
        else:
            other_synth_dir_config = None

        return other_synth_dir_config

    @staticmethod
    def get_root_dirpath(experiment_name: str, output_dir_name: str) -> Path:
        work_dir = Path(os.environ.get("WORK_DIR"))
        root_output_dir: Path = work_dir / "output" / experiment_name / output_dir_name
        return root_output_dir
