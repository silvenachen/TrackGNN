# TrackGNN: A Highly Parallelized and FIFO-Balanced Accelerator for the Particle-Track-Classification GNN on FPGAs
welcome to the open-sourced repository for TrackGNN!
## Overview
This project focuses on developing a highly efficient accelerator to process collision data and solve **track reconstruction** tasks from high-energy physics experiments. Our accelerator is specifically tailored for **FPGA** platforms, with optimized, **FIFO-Balanced parallel processing** of graph data obtained from an actual particle detector, **sPHENIX Experiment at RHIC**.
The TrackGNN accelerator implements two models with the same GNN architecture but different configurations. 

![TrackGNN Architecture](images/TrackGNN.jpg)
### Background and Model Architecture

The figure above illustrates the architecture of the GNN model used for track segmentation, which consists of three main network components: **InputNet**, **EdgeNet**, and **NodeNet**. This architecture processes a graph representing hits in a particle detector and performs both node and edge classification tasks to identify true particle tracks.

- **InputNet**: The input layer processes raw input features for each node (hit) in the graph, transforming them into a higher-dimensional embedding.
- **EdgeNet**: After the node embeddings are created, the EdgeNet processes pairs of nodes to predict whether a pair of hits is connected (i.e., part of the same track) or spurious.
- **NodeNet**: The NodeNet updates the node embeddings using aggregated information from neighboring nodes and their edges. This helps the model learn complex patterns and dependencies between hits.

The bottom section of the diagram illustrates the internal operations within the **EdgeNet**, which consists of a sequence of Multi-Layer Perceptron (MLP) layers and non-linear activation functions. These MLPs process pairs of node embeddings, refining the edge classification. The final layer applies a **sigmoid activation** to output the classification score for each edge, where a score closer to 1 indicates a true track connection, and a score closer to 0 indicates a spurious connection.

This design allows for efficient processing of particle hits and tracks, enabling accurate classification of both nodes (hits) and edges (connections between hits) in large-scale graph data from particle detectors.
