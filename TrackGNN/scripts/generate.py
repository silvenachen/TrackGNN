# import packages
import os
import shutil
from generate_activation_rom import generate_activation_rom as ga
from generate_wandb_rom import generate_wandb_rom as gw
from generate_graph import generate_graph as gg
from generate_config import generate_config as gc

SIM, EMU = True, False
TANH, SIGM = 'tanh', 'sigmoid'

# Hardware configurations
# Whether to simulate or emulate.
SIM_OR_EMU = SIM
# Parallelism parameters.
NODE_UNI_PARA = 4
EDGE_UNI_PARA = 8
DIM_PARA = 2
# Fraction width of data type.
DT_FRAC_WID = 8
# Number of graphs (less than 1024)
NUM_OF_GRAPHS = 100
# Number of layers.
NUM_OF_LAYERS = 1
# Activation function.
ACT_FUNC = TANH
# FIFO depth.
FIFO_DEPTH = 8

# Generate files
print('INFO: Start generating files.')
SIGM_TAIL, SIGM_SIZE, TANH_TAIL, TANH_SIZE = \
    ga(DT_FRAC_WID, NODE_UNI_PARA, EDGE_UNI_PARA, DIM_PARA)
NODE_FEA_DIM, NODE_EMB_DIM, MAX_DATA_OF_WANDB = \
    gw()
NODE_NUM_MAX, EDGE_NUM_MAX, MAX_DATA_OF_CPT = \
    gg(NUM_OF_GRAPHS, NUM_OF_LAYERS)
gc(
    SIM_OR_EMU, ACT_FUNC,
    NODE_NUM_MAX, EDGE_NUM_MAX,
    DT_FRAC_WID,
    MAX_DATA_OF_WANDB, MAX_DATA_OF_CPT,
    NODE_FEA_DIM, NODE_EMB_DIM,
    NODE_UNI_PARA, EDGE_UNI_PARA, DIM_PARA,
    SIGM_TAIL, SIGM_SIZE, TANH_TAIL, TANH_SIZE,
    NUM_OF_LAYERS, FIFO_DEPTH)
