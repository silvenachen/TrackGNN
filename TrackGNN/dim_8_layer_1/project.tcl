open_project workDict
set_top kernel_compute_graph
add_files kernel.cpp
add_files config.h
add_files -tb host.cpp
open_solution "solution1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 10 -name default
csim_design
#csynth_design
#cosim_design
exit