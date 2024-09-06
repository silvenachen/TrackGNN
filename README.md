# TrackGNN: A Highly Parallelized and FIFO-Balanced Accelerator for the Particle-Track-Classification GNN on FPGAs
welcome to the open-sourced repository for TrackGNN!
## Overview
This project focuses on developing a highly efficient accelerator to process collision data and solve **track reconstruction** tasks from high-energy physics experiments. Our accelerator is specifically tailored for **FPGA** platforms, with optimized, **FIFO-Balanced parallel processing** of graph data obtained from an actual particle detector, **sPHENIX Experiment at RHIC**.

### Background and Model Architecture
The TrackGNN accelerator implements two models with the same GNN architecture but different configurations. Each model processes input features of size 5 for each node, but they differ in embedding size and the number of layers. For the first model, the input network transforms the node embeddings to a dimension of 64, and the model has a total number of 4 layers. In the second optimized version, the node embedding size is reduced to 8 dimensions, and the model only requires 1 layer to obtain segmentation results.
![TrackGNN Architecture](image/model.jpg)
The above figure illustrated the architecture of the GNN model, whcih consists of three main network components: **InputNet**, **EdgeNet**, and **NodeNet**. The pixels in a particle detector are first grouped as hits and a graph is constructed on the hits. The model operates on graphs and alternates between EdgeNetwork and NodeNetwork layers to perform segment classification. The classification results can be further utilized by downstream track reconstruction tasks. Each component of the model is implemented as a Multi-Layer Perceptron (MLP) with tanh activations

- **InputNetwork**: The input layer processes raw input features for each node (hit) in the graph, transforming them into a higher-dimensional node embedding.
- **EdgeNetwork**: The EdgeNetwork evaluates the connections between node embeddings, predicting whether a pair of hits is part of the same track or a spurious pair. It uses a sigmoid activation in the final layer to output a score for each edge, with scores closer to 1 indicating a valid track connection and scores closer to 0 suggesting a false connection. A threshold of 0.5 is typically applied to classify the edges.
- **NodeNetwork**: The NodeNetwork updates the embeddings of each node by aggregating information from its neighboring nodes and the edges connecting them. 

