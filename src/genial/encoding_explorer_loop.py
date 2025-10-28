# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from argparse import ArgumentParser
import traceback

from loguru import logger
from typing import Any
import json
from time import time


from genial.config.config_dir import ConfigDir
from genial.experiment.task_generator import DesignGenerator
from genial.experiment.task_launcher import Launcher
from genial.experiment.task_analyzer import Analyzer
from genial.experiment.task_recommender import EncodingRecommender
from genial.training.mains.trainer_enc_to_score_value import EncToScoreTrainer

from genial.experiment.loop_module import LoopModule

from genial.config.logging import Logging as logging

from genial.utils.utils import load_database

all_configs = {
    "debug": {
        "experiment_name": "multiplier_2bi_4bo_permuti_allcells_unsigned_notech_fullsweep_only",
        "synth_version": 0,
        "init_nb_to_generate": 5,
        "device": 2,  # Trainer & Recommender
        "score_type": "trans",  # Trainer
        "score_rescale_mode": "minmax",  # Trainer
        "max_epochs": 2,  # Trainer
        "check_val_every_n_epoch": 1,  # Trainer
        "nb_new_designs": 5,  # Recommender
        "keep_percentage": 25.0,  # Recommender
        "split_ratios": [3.0 / 5, 1.0 / 5, 1.0 / 5],  # Trainer
        "max_epochs_p_iter": 1,  # Trainer
        "send_email": False,
        "synth_only": True,
        "skip_swact": True,
    },
    "mult4b_trans_minmax": {
        "experiment_name": "multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only",
        "synth_version": 0,
        "init_nb_to_generate": 40000,
        "score_type": "trans",  # Trainer
        "score_rescale_mode": "minmax",  # Trainer
        "check_val_every_n_epoch": 5,  # Trainer
        "nb_new_designs": 1000,  # Recommender
        "keep_percentage": 10.0,  # Recommender
        "split_ratios": [0.975, 0.0125, 0.0125],  # Trainer
        "max_epochs": 3000,
        # "output_dir_name":"2024-09-03_14-20_255a46e_SYNTHV0",
    },
    "mult4b_trans_standardize": {
        "experiment_name": "multiplier_4bi_8bo_permuti_allcells_notech_fullsweep_only",
        "synth_version": 0,
        "init_nb_to_generate": 400000,
        "score_type": "trans",  # Trainer
        "score_rescale_mode": "standardize",  # Trainer
        "check_val_every_n_epoch": 1,  # Trainer
        "nb_new_designs": 20000,  # Recommender
        "keep_percentage": 10.0,  # Recommender
        "split_ratios": [0.975, 0.0125, 0.0125],  # Trainer
        "max_epochs": 1000,
    },
    "mult4b_unsigned_trans_minmax": {
        "experiment_name": "multiplier_4bi_8bo_permuti_allcells_unsigned_notech_fullsweep_only",
        "synth_version": 0,
        "init_nb_to_generate": 10000,
        "device": 2,  # Trainer & Recommender
        "score_type": "trans",  # Trainer
        "score_rescale_mode": "minmax",  # Trainer
        "check_val_every_n_epoch": 5,  # Trainer
        "nb_new_designs": 1000,  # Recommender
        "keep_percentage": 1.0,  # Recommender
        "split_ratios": [0.975, 0.0125, 0.0125],  # Trainer
        "max_epochs": 3000,
        "max_epochs_p_iter": 200,  # Trainer
        # "output_dir_name":"2024-09-03_14-20_255a46e_SYNTHV0",
    },
}


class EncodingExplorer(LoopModule):
    """
    Main class for running an automated encoding exporation.
    It automatically detecteds whether initialization has sucessfully been run.
    The conditions for this are:
     - set `output_dir_name`
     - the file `<experiemnt_name>/<output_dir_name>/loop_init_done.token` exists
    In which case, the mode lversion number is read from the file `loop_config.json`
    Note that the later has been stored in the iter_mode configuration.
    Hence, task_analyzer and launchers will be run in continue/restart mode, not from scratch.
    """

    # Default configuration dictionnary
    config_dict = {
        "run_from_loop": True,
        "output_dir_name": None,
        "debug": False,
        "nb_workers": 128,
        "design_number_list": None,
        "existing_encodings_path": None,
        "only_gener": False,
        "do_gener": False,
        "skip_synth": False,
        "synth_only": False,
        "skip_swact": False,
        "skip_already_synth": True,
        "skip_already_swact": True,
        "restart": False,  # Launcher
        "rebuild_db": True,  # Analyzer
        "continue": False,  # Analyzer
        "skip_plots": False,  # Analyzer
        "fast_plots": True,  # Analyzer
        "skip_fullsweep_analysis": False,
        "skip_tests_list": [],
        "device": 5,  # Trainer & Recommender
        "restart_version": None,  # Trainer
        "seed": 626,  # Trainer
        "fast_dev_run": False,  # Trainer
        "trainer_version_number": None,  # Recommender & Trainer (for restarting)
        "dry_run": False,
        "score_type": "trans",  # Trainer
        "score_rescale_mode": "minmax",  # Trainer
        "max_epochs": 2000,  # Trainer
        "max_epochs_p_iter": 100,  # Trainer
        "batch_size": 512,  # Trainer & Recommender,
        "recommender_mode": "score_predictor",  # Recommender
        "send_email": True,
    }

    def check_configuration(self, config_dict: dict[str, Any], args_dict: dict[str, Any]):
        if args_dict.get("output_dir_name", None) is None:
            self.do_init_gener = True
            self.do_init = True
            logger.info(f"Output dir name has not been specified, initial generation will be done.")
        else:
            self.do_init_gener = False
            logger.info(f"Output dir name has been specified, initial generation will be skipped.")

            if config_dict.get("skip_init", False):
                self.do_init = False
                assert config_dict.get("output_dir_name", None) is not None, (
                    f"In order to skip initialization, you must provide which output directory to use."
                )
                logger.info(f"Initialization will be skipped.")
                logger.info(f"Output dir name has been set to: {config_dict.get('output_dir_name')}")

                if not config_dict.get("is_control_run", False):
                    # If control mode, not trainer needs to be up and running
                    if not self.init_checkpoint_token_filepath.exists():
                        assert config_dict.get("trainer_version_number", None) is not None, (
                            f"In order to skip initialization, you must provide which model to use for first iteration of recommender."
                        )
                    logger.info(f"Trainer version number has been set to: {config_dict.get('trainer_version_number')}")

            elif self.init_checkpoint_token_filepath.exists():
                self.do_init = False
            else:
                self.do_init = True

        if config_dict.get("synth_only"):
            if not config_dict.get("skip_swact"):
                err_msg = f"`synth_only` has been set to True, but `skip_swact` is False. Please fix this in the loop configuration used."
                logger.error(err_msg)
                raise ValueError(err_msg)

    def __init__(self, args_dict: dict):
        # launcher_args_dict = Launcher.parse_args()
        # analyzer_args_dict = Analyzer.parse_args()
        # trainer_args_dict = EncToScoreTrainer.parse_args()
        # recommender_args_dict = EncodingRecommender.parse_args()

        # Merge all argument dictionnaries
        # args_dict = launcher_args_dict | analyzer_args_dict | trainer_args_dict | recommender_args_dict

        # Old trainer version number is updated only when a new trainer version number is given in command line arguments

        # Make the Dir Config for this Loop
        self.config_dict.update(args_dict)
        self.dir_config = ConfigDir(**self.config_dict)
        logging().init_logging(self.dir_config.root_output_dir, mode="loop")

        # Update output dir name in self config_dict
        self.config_dict["output_dir_name"] = self.dir_config.output_dir_name
        self.init_checkpoint_token_filepath = self.dir_config.root_output_dir / "loop_init_done.token"
        self.loop_config_filepath = self.dir_config.root_output_dir / "loop_config.json"

        # Check loop configuration
        self.check_configuration(self.config_dict, args_dict)

        # Check whether loop initialization must be done
        if not self.do_init:
            if args_dict.get("trainer_version_number", None) is not None:
                self.change_model_version(args_dict.get("trainer_version_number", None))

            self.config_dict = json.load(open(self.loop_config_filepath, "r"))

            self.switch_config_dict_to_iter_mode()
            self.dir_config = ConfigDir(**self.config_dict)
            trainer_version_number = self.config_dict.get("trainer_version_number")
            self.dir_config.setup_trainer_version_output_dirs(trainer_version_number)
        else:
            if args_dict.get("skip_init_analysis", False):
                self.config_dict.update({"rebuild_db": False, "continue": True})
                self.dir_config = ConfigDir(**self.config_dict)
            if args_dict.get("skip_init_launching", False):
                self.config_dict.update(
                    {
                        "skip_already_synth": True,
                        "skip_already_swact": True,
                    }
                )
            trainer_version_number = self.config_dict.get("trainer_version_number", None)

        # Initialize all modules
        logger.info(f"Loop configuration has successfully been checked and done. Initializing all modules ...")
        self.design_generator = DesignGenerator(dir_config=self.dir_config)
        self.task_launcher = Launcher(dir_config=self.dir_config, nb_workers=int(self.config_dict["nb_workers"] / 2))
        self.task_analyzer = Analyzer(dir_config=self.dir_config)

        self.task_trainer = EncToScoreTrainer(
            analyzer=self.task_analyzer, trainer_version_number=trainer_version_number
        )
        self.config_dict["trainer_version_number"] = self.task_trainer.trainer.logger.version
        self.dir_config.setup_trainer_version_output_dirs()

        if self.config_dict.get("is_control_run", False):
            logger.info(
                f"`is_control_run` given in command line argument. The Recommender will be initiated in `control` mode."
            )
            self.recommender_mode = "random"
        else:
            self.recommender_mode = "score_predictor"
        self.task_recommender = EncodingRecommender(analyzer=self.task_analyzer, mode=self.recommender_mode)

        # Setting iter_nb to None to ensure init loop is not skipped.
        self.current_iter_nb = None
        logging().init_logging(self.dir_config.root_output_dir, mode="loop")

        # res_dict = launcher.run_from_scratch()
        # report_lines = launcher.build_report(res_dict=res_dict)

    def change_model_version(self, trainer_version_number: int) -> None:
        """
        This function updates the loop configuration file with the new model version.
        """

        # 1. Change model version in config file
        config_dict = json.load(open(self.loop_config_filepath, "r"))
        config_dict["trainer_version_number"] = trainer_version_number
        json.dump(config_dict, open(self.loop_config_filepath, "w"))

        return None

    def checkpoint_init_done(self):
        """Make the init_done token and store the config_dict"""
        # assert self.current_iter_nb == 0, f"The user should first switch to iteration mode before checkpointing Loop initialization."
        if not self.init_checkpoint_token_filepath.exists():
            json.dump(self.config_dict, open(self.loop_config_filepath, "w"))
            self.init_checkpoint_token_filepath.touch()
            logger.info(f"Loop initialization checkpointing done. Loop configuration file written in:")
            logger.info(f"{self.loop_config_filepath}\n\n\n")
        else:
            logger.info(
                f"Loop initialization checkpointing done has been skipped because the checkpoint token file has been found."
            )
            logger.info(
                f"You can enforce checkpointing to be done (and thus enforce overwriting the previous loop configuration file) by manually removing the token file {self.init_checkpoint_token_filepath}."
            )

    def switch_config_dict_to_iter_mode(self) -> None:
        self.config_dict.update(
            {
                "skip_already_synth": True,
                "skip_already_swact": True,
                "rebuild_db": False,
                "continue": True,
                "restart": True,
                "restart_version": True,
                "finetune": True,
            }
        )

    def switch_to_iter_mode(self, config_dict: dict[str, Any] | None = None, **kwargs) -> None:
        """
        Helper function that ensures that all modules run in itermode.
        Used to avoid erasing data.
        """
        logger.info(f"Switching to Loop iteration mode ...")
        # assert self.current_iter_nb is None, "The user must run an init loop prior to switching all modules to iteration mode."
        self.current_iter_nb = 0
        self.switch_config_dict_to_iter_mode()

        self.design_generator.switch_to_iter_mode(self.config_dict)
        self.task_launcher.switch_to_iter_mode(self.config_dict)
        self.task_analyzer.switch_to_iter_mode(self.config_dict)

        self.task_trainer.switch_to_iter_mode(self.config_dict, analyzer=self.task_analyzer)
        self.task_recommender.switch_to_iter_mode(
            self.config_dict, trainer_version_number=self.task_trainer.trainer.logger.version
        )
        logger.info(f"All modules have been successfully switched to Loop iteration mode.\n\n")

    def init_step(self):
        start_time = time()
        try:
            # Generation
            logger.info(f"Running Loop initialization ...")
            if self.do_init_gener:
                genered_design_list = self.design_generator.main(nb_to_generate=self.config_dict["init_nb_to_generate"])
            else:
                genered_design_list = []

            if not self.config_dict.get("skip_init_launching", False):
                # Launcher
                self.task_launcher.main()
            else:
                logger.info("Initial launch job have been skipped. (Received `skip_init_launching` argument.)")

            if not self.config_dict.get("skip_init_analysis", False):
                # Analyzer
                self.task_analyzer.main()
            else:
                logger.info("Initial analyze job have been skipped. (Received `skip_init_analysis` argument.)")

            # Trainer
            self.task_trainer.setup_datasets(analyzer=self.task_analyzer)
            if self.recommender_mode != "random":
                self.task_trainer.main()
            else:
                self.task_trainer.generate_n_save_report(iter_checkpoint_only=True)

            end_time = time()

            logger.info(f"All Loop initialization tasks successfully finished.")
            logger.info(f"{len(genered_design_list)} have been generated, launched and analyzed, and trained on.")
            logger.info(f"Loop initialization done in {(end_time - start_time) / 60 / 60:.2f}h.\n")

            status = "Success"
            error_msg = ""
        except Exception:
            status = "Failed"
            error_msg = traceback.format_exc()
        end_time = time()

        logger.info(error_msg)
        self.send_email(
            config_dict=self.config_dict,
            start_time=start_time,
            status=status,
            error_message=error_msg,
            calling_module="LooperInit",
            root_output_dir=self.dir_config.root_output_dir,
        )

        logger.info(f"Loop iteration {self.current_iter_nb} done | Status: {status}.")
        logger.info(
            f"It took {(end_time - start_time) / 60 / 60}h to run on {len(genered_design_list)} new generated paths.\n\n\n"
        )

        if status == "Failed":
            logger.error(f"Exiting due to error {error_msg}")
            raise RuntimeError()

        self.switch_to_iter_mode()
        self.checkpoint_init_done()

    def loop_step(self):
        """
        Standard loop function.
        In order, runs:
         - The recommender
         - The launcher
         - The analyzer
         - The trainer
        """

        start_time = time()
        try:
            do_recommend = self.update_iter_count()
            logger.info(f"Running loop iteration {self.current_iter_nb} ...")

            # Recommender (& Generation)
            recommended_paths = []
            if do_recommend:
                self.task_recommender.prepare_next_iter(strategy="latest_iter_best_golden_acc")
                recommended_paths = self.task_recommender.main_suggest(design_generator=self.design_generator)
                if len(recommended_paths) == 0:
                    logger.warning(f"Recommender did not recommend any design. Exiting.")
                    exit()
                # self.update_iter_count_file()
            else:
                pass

            # Launcher
            self.task_launcher.prepare_next_iter()
            self.task_launcher.main()

            # Analyzer
            self.task_analyzer.prepare_next_iter()
            self.task_analyzer.main()

            # Trainer
            self.task_trainer.prepare_next_iter(
                analyzer=self.task_analyzer,
                max_epochs_p_iter=self.config_dict.get("max_epochs_p_iter", 100),
                strategy="latest_iter_best_golden_acc",
            )
            # logger.info(f"Trainer iteration prepared, max epochs for this iter is:{self.config_dict.get("max_epochs_p_iter")}")
            if self.recommender_mode != "random":
                self.task_trainer.main()

            else:
                self.task_trainer.generate_n_save_report(iter_checkpoint_only=True)

            status = "Success"
            error_msg = ""
        except Exception:
            status = "Failed"
            error_msg = traceback.format_exc()

        end_time = time()

        logger.info(error_msg)
        self.send_email(
            config_dict=self.config_dict,
            start_time=start_time,
            status=status,
            error_message=error_msg,
            calling_module="Looper",
            root_output_dir=self.dir_config.root_output_dir,
        )

        logger.info(f"Loop iteration {self.current_iter_nb} done | Status: {status}.")
        logger.info(
            f"It took {(end_time - start_time) / 60 / 60}h to run on {len(recommended_paths)} new recommended paths.\n\n\n"
        )
        if status == "Failed":
            logger.error(f"Exiting due to error {error_msg}")
            raise RuntimeError()

    def update_iter_count(self) -> None:
        """
        Checks which iteration count to set.
        For this, it looks at the recommendation and training databases,
            and compares the iteration count they saved.
        It decides whether or not to do a new recommendation based on this.
        And it sets the iteration counter of all LoopModule children.
        """

        # Get maximum saved recommendation iteration number
        recom_db_path = self.task_recommender.get_recom_db_path()
        max_recom_iter_count = 0
        if recom_db_path.exists():
            recom_df = load_database(recom_db_path)

            # EncodingRecommender.fix_recom_db_iter_counts(recom_df=recom_df)

            max_recom_iter_count = recom_df["recom_iter_nb"].max()
            logger.info(f"Recom DB max iter_count is: {max_recom_iter_count}")

        # Get maximum saved training iteration number
        train_db_path = self.task_trainer.get_tain_db_path()
        if train_db_path.exists():
            train_df = load_database(train_db_path)
            max_train_iter_count = train_df["train_iter_nb"].max()
            logger.info(f"Recom DB max iter_count is: {max_recom_iter_count}")
        else:
            error_msg = f"Loop Initialization has not been run properly. Maybe the traininer failed? Or the training database was moved? Or the trainer_version_number is not correctly set in the `loops_config.json` file? Check the above lines to find the trainer_version_number and associated max iteration number written in the recom and train db."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Decide what to do
        if max_recom_iter_count > max_train_iter_count:
            # Recommendation is one step ahead of training
            # i.e., latest recommendation step ended successfully.
            # But not latest training step.
            # So: do not recommend
            do_recommend = False
            current_iter_count = max_recom_iter_count
        elif max_recom_iter_count == max_train_iter_count:
            # Recommendation and training are aligned, we can start a new iteration with a recommendation
            do_recommend = True
            current_iter_count = max_train_iter_count + 1  # Increase by one, we start a new iter
        else:
            raise ValueError(
                f"Recommendation loop iteration is not aligned with trainer loop iteration. Initialization probably went badly? Or trainer_version_numebr was never set in the recommender."
            )

        super().set_iter_count(current_iter_count)
        for attr in dir(self):
            if not attr.startswith("__"):
                attribute = getattr(self, attr)
                if isinstance(attribute, LoopModule):
                    attribute.set_iter_count(current_iter_count)

        logger.info(f"Loop iteration count have been updated to {current_iter_count}")

        return do_recommend

    def main(self):
        if self.do_init:
            self.init_step()

        self.switch_to_iter_mode()

        if not self.do_init and self.config_dict.get("skip_init", False):
            logger.warning(
                f"Init has been manually skipped with command line argument `skip_init`. If it does not exists, the configuration file will be written. If it exists, it won't be overwritten."
            )
            self.checkpoint_init_done()

        while True:
            self.loop_step()


def main_cli():
    arg_parser = ArgumentParser(description="All args here sould be used for quick debugging.")
    # arg_parser.add_argument("--nb_workers", type=int, help="nb_workers to use for this session")

    arg_parser.add_argument("--device", type=int, help="Override device number to use for training")
    arg_parser.add_argument("--skip_plots", action="store_true", help="Skip plots in analyzer")
    arg_parser.add_argument("--config", type=str, default="debug", help="Name of the configuration to use.")
    arg_parser.add_argument("--output_dir_name", type=str, default=None, help="Which output directory to start from.")
    arg_parser.add_argument(
        "--skip_init",
        action="store_true",
        help="Overriding parameter to help user restart an experiment without doing initialization. Note: in this case, the user must make sure that the training has already been done and specify the version to use with the `trainer_version_number` argument.",
    )
    arg_parser.add_argument(
        "--trainer_version_number",
        type=int,
        default=None,
        help="Version of the model to use for first iteration of training or recommendation. This has precendence over the cloop config stored by checkpointing. So, use with care.",
    )
    arg_parser.add_argument(
        "--is_control_run",
        action="store_true",
        help="Overriding parameter to help user restarting an experiment without doing initialization.",
    )
    arg_parser.add_argument(
        "--skip_init_analysis",
        action="store_true",
        help="Do not do initialization analyze job. Useful for starting from already existing data and doing first training.",
    )
    arg_parser.add_argument(
        "--skip_init_launching",
        action="store_true",
        help="Do not do initialization launch jobs. Useful for starting from already existing data and doing first training.",
    )
    arg_parser.add_argument("--yml_config_path", type=str, default=None, help="Specify which configuration to load.")

    args = arg_parser.parse_known_args()[0]
    config = all_configs[args.config]
    config.update(vars(args))

    # Check argument configuration

    genial = EncodingExplorer(config)
    genial.main()

    logger.info("All done.")


if __name__ == "__main__":
    main_cli()
