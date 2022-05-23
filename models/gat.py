
from typing import List
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool, global_mean_pool


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels: List[int], num_classes, edge_dim=None, num_conv=3, pooling="mean"):
        super(GAT, self).__init__()
        conv_h, lin_h = hidden_channels
        heads = num_conv  # implicitly
        edge_dim = edge_dim
        self.conv1 = GATConv(num_node_features, conv_h, heads, edge_dim=edge_dim, dropout=0.6) # edge_dim?
        self.conv2 = GATConv(conv_h * heads, lin_h, heads=1, concat=False, edge_dim=edge_dim, dropout=0.6)

        self.lin = Linear(lin_h, num_classes)
        self.pooling = pooling

    def pool(self, x, batch):
        if self.pooling == "max":
            return global_max_pool(x, batch)
        elif self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "add":
            return global_add_pool(x, batch)
        raise ValueError("Wrong pooling strategy")

    def forward(self, x, edge_index, batch, edge_attr=None, **kwargs):

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)

        x = self.pool(x, batch)
        x = self.lin(x)

        return x
