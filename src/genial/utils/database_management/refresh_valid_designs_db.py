from genial.experiment.file_parsers import __get_valid_designs_db

from genial.experiment.task_analyzer import Analyzer
from genial.globals import global_vars

import argparse
from pathlib import Path
from loguru import logger


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--root_dirpath", type=str, required=True)
    arg_parser.add_argument("--bulk_flow_dirname", type=str, required=False, default=None)
    args = arg_parser.parse_args()

    root_dirpath = Path(args.root_dirpath)
    global_vars["keep_not_valid"] = True

    logger.info(f"Root dirpath: {root_dirpath}")
    logger.info(f"Bulk flow dirname: {args.bulk_flow_dirname}")
    prompt = f"Continue with this configuration? (Please type in Yes or No)"
    logger.info(prompt)
    answer = input(prompt)

    if answer.lower().startswith("y"):
        logger.info("Continuing...")
    else:
        logger.info("Exiting...")
        exit(0)

    # Erase the existing database
    valid_db_filepath = root_dirpath / "valid_designs.db.pqt"
    if valid_db_filepath.exists():
        valid_db_filepath.unlink()
    else:
        logger.info(f"Valid designs database not found at {valid_db_filepath}")

    for step in Analyzer.__existing_steps__:
        db = __get_valid_designs_db(root_dirpath=root_dirpath, step=step, bulk_flow_dirname=args.bulk_flow_dirname)
    return db


if __name__ == "__main__":
    db = main()
