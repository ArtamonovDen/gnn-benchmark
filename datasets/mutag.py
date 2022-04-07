import csv
from pathlib import Path
import pandas as pd
import networkx
import os
import torch
import shutil
import igraph as ig
from torch_geometric.utils.convert import from_networkx
from typing import Dict, Union, List, Optional, Tuple, Callable

from torch_geometric.data import InMemoryDataset

from igraph_dataset import IGraphDataset


class MutagDataset(IGraphDataset):
    def __init__(self, root, label_path, transform=None, pre_transform=None):
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
        self.convert_raw_graphs()


root = "/home/friday/projects/hse_gnn/Netpro2vec/data/Mutag"
label_path = os.path.join(root, "Mutag.txt")
g = MutagDataset(root, label_path)
