import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_add_pool, global_mean_pool


class GraphGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GraphGCN, self).__init__()
        # torch.manual_seed(42)
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.conv5 = GraphConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        # TODO: normalize by batch?
        if edge_weight is not None:
            edge_weight = torch.flatten(edge_weight)
            edge_weight = F.normalize(edge_weight, dim=0)
        # TODO: точно и edge_index edge weigt правильно отдаётся?
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = self.conv5(x, edge_index, edge_weight).relu()

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # x = global_add_pool(x, batch)

        x = self.lin1(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)
