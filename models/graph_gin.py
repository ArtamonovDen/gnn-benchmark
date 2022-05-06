import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_max_pool


class GIN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(num_node_features, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            )
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            )
        )

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, **kwargs):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
