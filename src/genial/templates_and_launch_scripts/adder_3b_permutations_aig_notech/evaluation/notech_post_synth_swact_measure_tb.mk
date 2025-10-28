# SPDX-License-Identifier: BSD-3-Clause Clear
# Copyright (c) 2024-2025 HUAWEI.
# All rights reserved.
#
# This software is released under the BSD 3-Clause Clear License.
# See the LICENSE file in the project root for full license information (https://github.com/huawei-csl/GENIAL).

#file: Makefile
CWD=$(shell pwd)

TOPLEVEL_LANG ?= verilog
SIM ?= verilator
WAVES ?= 0
# PLUSARGS="+post_synthesis=1"

#Paths to HDL source files
ifeq ($(TOPLEVEL_LANG),verilog)
  VERILOG_SOURCES =$(CWD)/../hdl/mydesign_synth_wrapper.v
  VERILOG_SOURCES +=$(CWD)/../../notech_primitives.v
  VERILOG_SOURCES +=$(CWD)/../../../synth/out/mydesign_yosys.v
else
  $(error "A valid value (verilog) was not provided for TOPLEVEL_LANG=$(TOPLEVEL_LANG)")
endif

DUT      = mydesign_synth_wrapper     #module under test
TOPLEVEL = mydesign_synth_wrapper             #top module
MODULE := testbench   #python testbench file
# COCOTB_HDL_TIMEUNIT=1us        
# COCOTB_HDL_TIMEPRECISION=1us
COCOTB_HDL_TIMEPRECISION = 100ps

CUSTOM_SIM_DEPS=$(CWD)/Makefile

ifeq ($(SIM),questa)
    SIM_ARGS=-t 100ps
endif

ifeq ($(SIM),icarus)
    COMPILE_ARGS += -DFIXED_POINT  #Any extra arguments to the iverilog 
								#command can be placed here. Any parameters
								#in the verilog module can be overriden from 
                                   #here as well. However, they cannot be change
                                   #during runtime.
else ifeq ($(SIM),verilator)
    # COMPILE_ARGS += -DSIMULATION -DFUNCTIONAL
	  COMPILE_ARGS += -Wno-SELRANGE -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-MODDUP 

    COMPILE_ARGS += $(foreach v,$(filter PARAM_%,$(.VARIABLES)),-G$(subst PARAM_,,$(v))=$($(v)))

    ifeq ($(WAVES),1)
      EXTRA_ARGS += --trace --trace-structs  
    endif
endif
    # EXTRA_ARGS +=  --trace --trace-fst --trace-structs

# EXTRA_ARGS += --trace --trace-structs # A VCD file named dump.vcd will be generated in the current directory.
# EXTRA_ARGS += --trace-fst --trace-structs # The resulting file will be dump.fst and can be opened by gtkwave dump.fst.

include $(shell cocotb-config --makefiles)/Makefile.sim