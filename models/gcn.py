from typing import List
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels: List[int], num_classes, num_conv=3, pooling="max"):
        super(GCN, self).__init__()
        conv_h, lin_h = hidden_channels[0], hidden_channels[1]

        self.pooling = pooling
        self.convs = [GCNConv(num_node_features, conv_h)] + [GCNConv(conv_h, conv_h) for _ in range(num_conv-1)]
        self.lin1 = Linear(conv_h, lin_h)
        self.lin2 = Linear(lin_h, num_classes)

    def pool(self, x, batch):
        if self.pooling == "max":
            return global_max_pool(x, batch)
        elif self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "add":
            return global_add_pool(x, batch)
        raise ValueError("Wrong pooling strategy")

    def forward(self, x, edge_index, edge_weight, batch, **kwargs):
        # 1. Obtain node embeddings
        if edge_weight is not None:
            edge_weight = torch.flatten(edge_weight)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight).relu()

        # 2. Readout layer
        x = self.pool(x, batch)

        # 3. Apply a final classifier
        x = self.lin1(x).relu()
        x = self.lin2(x)

        return x
