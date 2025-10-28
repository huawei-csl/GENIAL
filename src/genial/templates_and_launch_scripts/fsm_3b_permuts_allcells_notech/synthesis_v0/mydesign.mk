# Copyright 2023 ETH Zurich and University of Bologna
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

BENDER	?= bender
MORTY 	?= morty
SVASE 	?= svase
SV2V  	?= sv2v
PYTHON3 ?= python3
REGGEN  ?= $(PYTHON3) $(shell $(BENDER) path register_interface)/vendor/lowrisc_opentitan/util/regtool.py

VLOG_ARGS += -suppress 2583 -suppress 13314 -svinputport=compat

# SAFED_ROOT 	 ?= $(shell $(BENDER) path safety_island)
MYDESIGN_ROOT    ?= $(shell $(BENDER) path mydesign_soc)
MYDESIGN_HW_DIR  ?= $(MYDESIGN_ROOT)/hw/mydesign/hdl


################
# Dependencies #
################
BENDER_ROOT ?= $(MYDESIGN_ROOT)/.bender

# Ensure both Bender dependencies and (essential) submodules are checked out
$(BENDER_ROOT)/.deps: $(SAFED_ROOT)/.deps
	cd $(MYDESIGN_ROOT) && git submodule update --init --recursive
	@touch $@

# Make sure dependencies are more up-to-date than any targets run
# adding this makes the checkout automatic
# ifeq ($(shell test -f $(BENDER_ROOT)/.deps && echo 1),)
# -include $(BENDER_ROOT)/.deps
# endif

## Checkout/update dependencies using Bender
checkout: $(BENDER_ROOT)/.deps

## reset dependencies (without updating Bender.lock)
clean-deps:
	rm -rf .bender
	cd $(MYDESIGN_ROOT) && git submodule deinit -f --all

.PHONY: checkout clean-deps


##########################
# Hardware Configuration #
##########################
# The bulk is done in hw/mydesign_pkg.sv
HW_ALL := 
SIM_ALL :=


# #############
# # Verilator #
# #############

# $(MYDESIGN_ROOT)/test/verilator/mydesign.f: Bender.lock
# 	mkdir -p $(MYDESIGN_ROOT)/test/verilator
# 	$(BENDER) script verilator -t rtl -t verilator -t verilator_test -DSYNTHESIS -DVERILATOR > $@

# SIM_ALL += $(MYDESIGN_ROOT)/test/verilator/mydesign.f

####################
# Open Source Flow #
####################
MYDESIGN_OUT       ?= $(MYDESIGN_ROOT)/pickle
TOP_DESIGN     ?= mydesign_comb
BENDER_TARGERS ?= asic ihp130 rtl synthesis notech
MORTY_DEFINES  ?= VERILATOR SYNTHESIS MORTY TARGET_ASIC TARGET_SYNTHESIS

# list of source files
$(MYDESIGN_OUT)/mydesign_sources.json: Bender.yml
	mkdir -p $(MYDESIGN_OUT)
	$(BENDER) sources -f $(foreach t,$(BENDER_TARGERS),-t $(t)) > $@

# pickle source files into one file/context
$(MYDESIGN_OUT)/mydesign_morty.sv: $(MYDESIGN_OUT)/mydesign_sources.json $(MYDESIGN_HW_DIR)/* $(HW_ALL)
	$(MORTY) -q -f $< -o $@ $(foreach d,$(MORTY_DEFINES),-D $(d)=1)

# simplify SystemVerilog by propagating parameters and unfolding generate statements
$(MYDESIGN_OUT)/mydesign_svase.sv: $(MYDESIGN_OUT)/mydesign_morty.sv
	$(SVASE) $(TOP_DESIGN) $@ $<
	sed -i 's/module $(TOP_DESIGN)__[[:digit:]]\+/module $(TOP_DESIGN)/' $@

# convert SystemVerilog to Verilog
$(MYDESIGN_OUT)/mydesign_sv2v.v: $(MYDESIGN_OUT)/mydesign_svase.sv
	$(SV2V) --oversized-numbers --write $@ $<

.PHONY: pickle
## Generate Safety Island verilog file for synthesis
pickle: $(MYDESIGN_OUT)/mydesign_sv2v.v

include ihp130/technology.mk
include synth/yosys.mk
include openroad/openroad.mk


###########
# PHONIES #
###########

.PHONY: hw-all
hw-all: $(HW_ALL)

.PHONY: sim-all
sim-all: $(SIM_ALL)
