from pathlib import Path
from loguru import logger

from genial.utils.utils import extract_int_string_from_string


class Logging:
    __log_handler__ = []

    def init_logging(self, log_dirpath: Path, mode: str = "trainer", reset: bool = False):
        log_filepath = log_dirpath / f"{mode}_log.log"

        # log_version = self._get_file_version(log_filepath)
        # log_filepath = log_filepath.with_name(f"{mode}_log_{log_version}.log")

        self._init_logging(log_filepath=log_filepath, reset=reset)

    def _get_file_version(self, log_filepath: Path):
        log_version = extract_int_string_from_string(log_filepath.name)
        if not log_version.isdigit():
            log_version = 0
        else:
            log_version = int(log_version) + 1
        return log_version

    def _init_logging(self, log_filepath: Path, reset: bool = False):
        if not log_filepath.exists():
            log_filepath.parent.mkdir(parents=True, exist_ok=True)
            log_filepath.touch()
        else:
            assert log_filepath.is_file(), f"log_filepath given is not a file, received: {log_filepath}"

        logger.info(f"Logging to {log_filepath}")
        if reset:
            logger.warning(f"Resetting logfile")
            log_filepath.write_text("")

        # Remove previous log file handler
        if len(self.__log_handler__) > 0:
            logger.remove(self.__log_handler__.pop())

        # Add new log file handler
        self.__log_handler__.append(logger.add(log_filepath, level="INFO", rotation="1 week", retention="10 days"))
        logger.info(f"Logging file set to {log_filepath}")
