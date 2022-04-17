from pathlib import Path
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

from datasets.igraph_dataset import IGraphDataset


class MutagDataset(IGraphDataset):
    def __init__(self, root, label_path, transform=None, pre_transform=None):
        self.node_encoder = OneHotEncoder()
        super().__init__(root, label_path, transform, pre_transform)

    @property
    def edge_attributes(self):
        return ["label"]

    @property
    def node_attributes(self):
        return ["label"]

    @property
    def raw_file_names(self):
        return sorted(
            Path(self.raw_dir).iterdir(), key=lambda m: int(m.stem.split("_")[-1])
        )

    def process(self):
        # Becasue of if 'process' in self.__class__.__dict__:
        # we need process() method defined in child class
        data, slices = self.convert_raw_graphs()

        # One-hot encodding node-features
        x = data.x
        x = self.node_encoder.fit_transform(x).toarray()
        x = torch.tensor(x)
        data.x = x.float()  # TODO: Not sure if it's okay to do that. but seems it works
        torch.save((data, slices), self.processed_paths[0])


root = "/home/friday/projects/hse_gnn/Netpro2vec/data/Mutag"
label_path = os.path.join(root, "Mutag.txt")
g = MutagDataset(root, label_path)
print(g)


"""
from torch_geometric.datasets import TUDataset
dataset = TUDataset(root="data/TUDataset", name="MUTAG")
>> dataset.data
Data(x=[3371, 7], edge_index=[2, 7442], edge_attr=[7442, 4], y=[188])
>>> g.data
# Data(x=[3371, 1], edge_index=[2, 3721], edge_attr=[3721, 1], y=[188])
Data(x=[3371, 7], edge_index=[2, 7442], y=[188]) # UPDATED
"""

# TODO: WHY?
"""
>>> dataset.data.is_undirected()
False
>>> g.data.is_undirected()
True
"""
