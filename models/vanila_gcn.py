from typing import List
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool


def load_model():
    return VanilaGCN(1, 2, 3)  # TODO


class VanilaGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels: List[int], num_classes):
        super(VanilaGCN, self).__init__()
        conv_h, lin_h = hidden_channels[0], hidden_channels[1]
        lin_h = 64
        self.conv1 = GCNConv(num_node_features, conv_h)
        self.lin1 = Linear(conv_h, lin_h)
        self.lin2 = Linear(lin_h, num_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings
        if edge_weight is not None:
            edge_weight = torch.flatten(edge_weight)
            # edge_weight = F.normalize(edge_weight, dim=0)
        x = x.float()  # TODO: for now
        x = self.conv1(x, edge_index, edge_weight).relu()
        # x = self.conv2(x, edge_index, edge_weight).relu()
        # x = self.conv3(x, edge_index, edge_weight).relu()

        # 2. Readout layer
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = global_max_pool(x, batch)

        # 3. Apply a final classifier
        x = self.lin1(x).relu()
        x = self.lin2(x)

        # return F.log_softmax(x, dim=-1) ?
        return x
