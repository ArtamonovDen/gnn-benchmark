import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool


class MLP(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(MLP, self).__init__()
        self.lin1 = Linear(num_node_features, hidden_channels[0])
        self.lin2 = Linear(hidden_channels[0], hidden_channels[1])
        self.lin3 = Linear(hidden_channels[1], hidden_channels[2])
        self.lin4 = Linear(hidden_channels[2], num_classes)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        x = self.lin3(x).relu()
        x = self.lin4(x)
        return x
