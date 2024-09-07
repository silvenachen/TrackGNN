############################## Help Section ##############################
ifneq ($(findstring Makefile, $(MAKEFILE_LIST)), Makefile)
help:
	@echo "Makefile Usage:"
	@echo "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	@echo "      Command to generate the design for specified Target and Shell."
	@echo ""
	@echo "  make build TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	@echo "      Command to build xclbin application."
	@echo ""
	@echo "  make program TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	@echo "      Command to program the FPGA with the generated xclbin."
	@echo ""
	@echo "  make host"
	@echo "      Command to build the host application."
	@echo ""
	@echo "  make clean"
	@echo "      Command to remove the generated non-hardware files."
	@echo ""
	@echo "  make cleanall"
	@echo "      Command to remove all the generated files."
	@echo ""

endif

############################## Setting up Project Variables ##############################
TARGET := hw
PLATFORM := xilinx_u280_gen3x16_xdma_1_202211_1

TEMP_DIR := ./_x.$(TARGET).$(PLATFORM)
BUILD_DIR := ./build_dir.$(TARGET).$(PLATFORM)

LINK_OUTPUT := $(BUILD_DIR)/kernel_compute_graph.link.xclbin
PACKAGE_OUT := ./package.$(TARGET)

############################## Setting up Kernel Variables ##############################
VPP_FLAGS := --save-temps

############################## Setting up Host Variables ##############################
HOST_CXX := g++
HOST_CXXFLAGS := -std=c++1y -DVITIS_PLATFORM=$(PLATFORM) -D__USE_XOPEN2K8 -I$(XILINX_XRT)/include/ -I/opt/xilinx/2022.1/Vitis_HLS/2022.1/include/ -I. -O2 -g -Wall -c -fmessage-length=0
HOST_LDFLAGS := -lxilinxopencl -lpthread -lrt -lstdc++ -L$(XILINX_XRT)/lib/ -Wl,-rpath-link,$(XILINX_XRT)/lib

HOST_OBJECTS := host.o xcl2.o
HOST_EXE := host_app

############################## Setting Targets ##############################
.PHONY: all clean cleanall host

all: build host

build: $(BUILD_DIR)/kernel_compute_graph.xclbin

$(TEMP_DIR)/kernel_compute_graph.xo: ./kernel.cpp config.h
	mkdir -p $(TEMP_DIR)
	v++ -c $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) -k kernel_compute_graph --temp_dir $(TEMP_DIR) -I. -o '$@' $<

$(BUILD_DIR)/kernel_compute_graph.xclbin: $(TEMP_DIR)/kernel_compute_graph.xo
	mkdir -p $(BUILD_DIR)	
	v++ -l $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) --temp_dir $(TEMP_DIR) -o '$(LINK_OUTPUT)' $^
	v++ -p $(LINK_OUTPUT) -t $(TARGET) --platform $(PLATFORM) --package.out_dir $(PACKAGE_OUT) -o $(BUILD_DIR)/kernel_compute_graph.xclbin

program: $(BUILD_DIR)/kernel_compute_graph.xclbin
	xbutil program --device $(PLATFORM) --bitstream $(BUILD_DIR)/kernel_compute_graph.xclbin

host: $(HOST_EXE)

$(HOST_EXE): $(HOST_OBJECTS)
	$(HOST_CXX) -o $@ $(HOST_OBJECTS) $(HOST_LDFLAGS)

host.o: host.cpp config.h
	$(HOST_CXX) $(HOST_CXXFLAGS) -o $@ $<

xcl2.o: xcl2.cpp xcl2.hpp
	$(HOST_CXX) $(HOST_CXXFLAGS) -o $@ $<

clean:
	-rm -rf $(TEMP_DIR) $(BUILD_DIR) *.log *.jou *.wcfg *.wdb $(HOST_OBJECTS) $(HOST_EXE)

cleanall: clean
	-rm -rf package.*
