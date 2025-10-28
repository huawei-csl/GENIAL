# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

from genial.config.config_dir import ConfigDir
from genial.experiment.task_analyzer import Analyzer


def setup_analyzer(**kwargs: dict) -> Analyzer:
    # Make sure that the analyzer is setup and do the correct checks
    assert "rebuild_db" in kwargs.keys(), (
        "Argument parser sould be initialized using the `analzer_parser` from `genial.config.arg_parsers`"
    )

    # Make sure the analyzer will load the databases (and not erase them)
    kwargs["rebuild_db"] = False
    kwargs["skip_synth"] = True
    kwargs["skip_swact"] = True

    dir_config = ConfigDir(is_analysis=True, **kwargs)

    analyzer = Analyzer(dir_config=dir_config, reset_logs=False, skip_log_init=True, read_only=True)
    analyzer.format_databases()
    analyzer.align_databases()

    return analyzer
