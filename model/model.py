import os
import torch
from torch_scatter import scatter_add
import torch.nn as nn

# InputNetwork: Handles the input feature transformation.
class InputNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InputNetwork, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.tanh(x)
        return x

# EdgeNetwork: Processes edges between nodes based on their embeddings.
class EdgeNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(EdgeNetwork, self).__init__()
        self.fc1 = nn.Linear(in_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        src, dst = edge_index
        edge_inputs = torch.cat([x[src], x[dst]], dim=1)
        x = self.tanh(self.fc1(edge_inputs))
        # print(x)
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)

# NodeNetwork: Updates node embeddings based on incoming/outgoing edge features.
class NodeNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NodeNetwork, self).__init__()
        self.fc1 = nn.Linear(in_dim * 3, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.fc4 = nn.Linear(out_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, x, e, edge_index):
        src, dst = edge_index
        # Aggregate edge-weighted incoming/outgoing features
        x_src, x_dst = x[src], x[dst]
        e_sq = e[:, None]
        y_src, y_dst = e_sq * x_src, e_sq * x_dst
        mi = scatter_add(y_src, dst, dim=0, dim_size=x.shape[0])
        mo = scatter_add(y_dst, src, dim=0, dim_size=x.shape[0])

        node_inputs = torch.cat([mi, mo, x], dim=1)
        print(node_inputs)

        x = self.tanh(self.fc1(node_inputs))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))

        return x


class GnnClassifier(nn.Module):
    """
    :param input_dim: Input feature dimension.
    :param hidden_dim: Hidden node/embedding dimension.
    :param num_layers: Number of GNN layers for message passing.
    """
    def __init__(self, input_dim, hidden_dim, num_of_layers):
        super(GnnClassifier, self).__init__()
        self.num_of_layers = num_of_layers
        self.input_network = InputNetwork(input_dim, hidden_dim)
        self.edge_network = EdgeNetwork(hidden_dim, hidden_dim)
        self.node_network = NodeNetwork(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.input_network(x)  # x (batch_size, hidden_dim)

        for n in range(self.num_of_layers + 1):
            e = self.edge_network(x, edge_index)
            if n == self.num_of_layers:
                break
            x += self.node_network(x, e, edge_index)


if __name__ == "__main__":
    input_dim = 5
    #configuration for a model with 8-dimensional embeddings and 1 layer
    hidden_dim = 8
    num_of_layer = 1

    #configuration for a model with 64-dimensional embeddings and 4 layer
    # hidden_dim = 64
    #     # num_of_layer = 4

    classifier = GnnClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_of_layers=num_of_layer)
    print("Model instantiated!")
