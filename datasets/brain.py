from pathlib import Path
import os
import torch
import igraph as ig
import pandas as pd

from datasets.igraph_dataset import IGraphDataset
from torch_geometric.data import Data


def get_dataset():
    root = "/home/friday/projects/hse_gnn/datasets/TumotMet/Brain"
    label_path = os.path.join(root, "Mutag.txt")
    return BrainDataset(root, label_path)


def ggg(root):
    df = pd.read_csv(f"{root}/sample_sheet.tsv", sep="\t")
    lablels_mapping = df[["Sample.ID", "diagnosis"]]
    lablels_mapping = lablels_mapping.set_index("Sample.ID")
    lablels_mapping = lablels_mapping["diagnosis"]
    # lablels_mapping['TCGA-32-1970-01A']


class BrainDataset(IGraphDataset):
    def __init__(self, root, label_path, transform=None, pre_transform=None):
        super().__init__(root, label_path, transform, pre_transform)

    @property
    def edge_attributes(self):
        return None

    @property
    def node_attributes(self):
        return None

    @property
    def raw_file_names(self):
        return Path(self.raw_dir).iterdir()

    def init_graph_labels(self):
        df = pd.read_csv(self.label_path, sep="\t")
        lablels_mapping = df[["Sample.ID", "diagnosis"]]
        lablels_mapping = lablels_mapping.set_index("Sample.ID")
        lablels_mapping = lablels_mapping[~lablels_mapping["diagnosis"].isna()]

        diagnosis_cat = self.label_encoder.fit_transform(
            lablels_mapping["diagnosis"].values  # TODO: check
        )
        lablels_mapping["diagnosis_cat"] = diagnosis_cat
        self.labels = lablels_mapping["diagnosis_cat"]

    def get_label_for_graph(self, graph_name):
        # TODO: add check if graph doesn't exists!
        if graph_name in self.labels:
            encoded_label = self.labels[graph_name]
            return torch.LongTensor(encoded_label)
        return None

    def get_node_attrs(self, iG: ig.Graph) -> torch.FloatTensor:
        """
        Use adj matrix as X
        """
        # TODO: add edj weights and use Laplasian
        adj = iG.get_adjacency().data
        adj = torch.FloatTensor(adj)
        print(f"Use adj matrix as X. Shape is {adj.shape}")
        return adj

    def process(self):
        graph_data_list = []
        for i, graph_path in enumerate(self.raw_file_names):
            print(f"{i}: processing {graph_path.resolve()}")

            graph_name = graph_path.stem.split("_")[1]  # remove meanSum prefix
            pyg_y = self.get_label_for_graph(graph_name)
            if pyg_y is None:
                print(f"Skip graph {graph_name} as it has no label")
                continue

            iG: ig.Graph = ig.load(graph_path)
            pyg_edge_index = self.get_edges_from_igraph(iG)
            pyg_x = self.get_node_attrs(iG)
            # Create Data object
            graph = Data(x=pyg_x, edge_index=pyg_edge_index, y=pyg_y)
            graph_data_list.append(graph)

        data, slices = self.collate(graph_data_list)
        torch.save((data, slices), self.processed_paths[0])


"""
from torch_geometric.datasets import TUDataset
dataset = TUDataset(root="data/TUDataset", name="MUTAG")
>> dataset.data
Data(x=[3371, 7], edge_index=[2, 7442], edge_attr=[7442, 4], y=[188])
>>> g.data
# Data(x=[3371, 1], edge_index=[2, 3721], edge_attr=[3721, 1], y=[188])
Data(x=[3371, 7], edge_index=[2, 7442], y=[188]) # UPDATED
"""
