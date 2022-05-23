
from typing import List
import torch
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv, global_add_pool, global_max_pool, global_mean_pool


class GCNII(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels: List[int], num_classes, num_conv=3, dropout=0.5, alpha=0.5, theta=0.1, pooling="mean"):
        super(GCNII, self).__init__()
        conv_h = hidden_channels[0]

        self.pooling = pooling

        self.lin1 = Linear(num_node_features, conv_h)
        self.lin2 = Linear(conv_h, num_classes)

        self.convs = ModuleList()
        for layer in range(num_conv):
            self.convs.append(
                GCN2Conv(conv_h, alpha, theta, layer + 1)
            )

        self.dropout = dropout

    def pool(self, x, batch):
        if self.pooling == "max":
            return global_max_pool(x, batch)
        elif self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "add":
            return global_add_pool(x, batch)
        raise ValueError("Wrong pooling strategy")

    def forward(self, x, edge_index, batch, edge_weight=None, **kwargs):

        if edge_weight is not None:
            edge_weight = torch.flatten(edge_weight)

        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lin1(x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, edge_index, edge_weight).relu()

        x = self.pool(x, batch)
        x = self.lin2(x)

        return x
