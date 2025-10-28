# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from typing import Any
import time
import sys
from pathlib import Path
import os
from genial.utils.utils import send_email as _send_email
import textwrap

from loguru import logger

from genial.utils.utils import extract_int_string_from_string


class LoopModule:
    """Base class for any module used in a loop."""

    def __init__(self) -> None:
        self.current_iter_nb = 0

    def set_iter_count(self, iteration_count: int) -> None:
        """Set the recommendation count to a specific value."""
        self.current_iter_nb = iteration_count

    def prepare_next_iter(self, **kwargs) -> bool:
        """
        Placeholder for prepare next iteration function.
        Returns:
            True if something went wrong.
            False otherwise.
        """
        raise NotImplementedError()

    def switch_to_iter_mode(self, config_dict: dict[str, Any], **kwargs) -> None:
        """
        Placeholder for switching to iteration mode function.
        """
        raise NotImplementedError()

    def send_email(
        self,
        config_dict: dict[str:Any],
        start_time: float,
        status: str,
        error_message: str,
        calling_module: str,
        root_output_dir: Path,
    ):
        def define_log_file_name():
            src_dir = Path(os.getenv("SRC_DIR"))
            email_logs_dir = src_dir / "z_email_logs" / calling_module
            if not email_logs_dir.exists():
                email_logs_dir.mkdir(exist_ok=True, parents=True)

            version_number = 0
            for file in email_logs_dir.iterdir():
                if file.is_file():
                    version_number = max(int(extract_int_string_from_string(file.name)), version_number)
            version_number += 1

            return version_number, email_logs_dir / f"log_v{version_number}.log"

        if config_dict.get("send_email", False):
            end_time = time.time()
            duration = end_time - start_time
            formatted_duration = time.strftime("%H:%M:%S", time.gmtime(duration))

            command = " ".join(sys.argv)

            # Prepare email body
            email_body = textwrap.dedent(f"""
            
            Job Information:

            - **Start Date**: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}
            - **Iteration Number**: {self.current_iter_nb}
            - **Duration**: {formatted_duration}
            - **Status**: {status}
            - **Error**: {error_message}
            - **Files**: {root_output_dir.relative_to(Path(os.getenv("WORK_DIR")))}
            - **Command Line**: {command}
            
            """)

            # Store e-mail message in a file
            log_version, log_filepath = define_log_file_name()
            log_filepath.write_text(email_body)

            # Prepare e-mail subject
            email_subject = f"{calling_module.title()} | log v{log_version} | {status}"

            email_body_short = textwrap.dedent(f"""
            
            Job Information:

            - **Start Date**: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}
            - **Iteration Number**: {self.current_iter_nb}
            - **Duration**: {formatted_duration}
            - **Status**: {status}
            - **More Info in Log Version**: {log_version}
            
            """)

            _send_email(email_subject, body=email_body_short, calling_function=calling_module)
        else:
            logger.warning(f"End job e-mail was not sent because argument `--send_email` was not set.")
