# Copyright (c) 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Authors:
# - Philippe Sauter <phsauter@ethz.ch>

# target clock-period in pico-seconds
export YOSYS_TARGET_PERIOD_PS := 2000

# modules Yosys will treat as blackboxes
export YOSYS_BLACKBOX_MODULES := "*tc_sram_blackbox*"

# flatten hierarchy (except for below selections)
export YOSYS_FLATTEN_HIER := 1

# a list of yosys selection strings, all selected instances will be 
# kept as a seperate hierarchical element, all others will be flattened
# https://yosyshq.readthedocs.io/projects/yosys/en/latest/cmd/select.html
export YOSYS_KEEP_HIER_INST :=  "t:mydesign_top*"

# the paths (full names) of all instances matching these strings is reported
# for floorplaning or writing constraints
export YOSYS_REPORT_INSTS :=	"t:RM_IHPSG13_*"

# export tech files to the environment so its easily available in yosys
export YOSYS_TECH_CELLS 	:= $(TECH_CELLS)
export YOSYS_TECH_MACROS	:= $(TECH_MACROS)
export YOSYS_TECH_TIEHI 	:= $(TECH_CELL_TIEHI)
export YOSYS_TECH_TIELO 	:= $(TECH_CELL_TIELO)