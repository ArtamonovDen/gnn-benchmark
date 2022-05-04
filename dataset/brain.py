from pathlib import Path
import os
from typing import List
from sklearn import preprocessing
import torch
import igraph as ig
import pandas as pd

from torch_geometric.data import Data, InMemoryDataset, Dataset
from dataset.igraph_tools import (
    get_degree_matrix,
    get_edges_from_igraph,
    get_edges_with_weghts_from_igraph,
)


def init_graph_labels(label_path, label_encoder):

    df = pd.read_csv(label_path, sep="\t")
    lablels_mapping = df[["Sample.ID", "diagnosis"]]
    lablels_mapping = lablels_mapping.set_index("Sample.ID")
    lablels_mapping = lablels_mapping[~lablels_mapping["diagnosis"].isna()]

    diagnosis_cat = label_encoder.fit_transform(lablels_mapping["diagnosis"].values)
    lablels_mapping["diagnosis_cat"] = diagnosis_cat
    labels = lablels_mapping["diagnosis_cat"]
    return labels


def get_label_for_graph(labels, graph_name):
    if graph_name in labels:
        encoded_label = labels[graph_name]
        return torch.tensor(encoded_label, dtype=torch.int64)
    return None


class TumorBrainInMemoryDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        label_path,
        transform=None,
        pre_transform=None,
        are_directed=False,
    ):
        self.are_directed = are_directed  # if graphs in dataset directed
        self.label_encoder = preprocessing.LabelEncoder()
        self.labels = init_graph_labels(label_path, self.label_encoder)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return Path(self.raw_dir).iterdir()

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "graphml")

    @property
    def processed_file_names(self) -> List[str]:
        return ["graph_data.pt"]

    def process(self):
        graph_data_list = []
        for i, graph_path in enumerate(self.raw_file_names):
            print(f"{i}: processing {graph_path.resolve()}")

            graph_name = graph_path.stem.split("_")[1]  # remove meanSum prefix
            y = get_label_for_graph(self.labels, graph_name)
            if y is None:
                print(f"Skip graph {graph_name} as it has no label")
                continue

            iG: ig.Graph = ig.load(graph_path)
            edge_index = get_edges_from_igraph(iG, self.are_directed)
            x = get_degree_matrix(iG)

            # Create Data object
            graph = Data(x=x, edge_index=edge_index, y=y)  # TODO: no weights yet!
            graph_data_list.append(graph)

        data, slices = self.collate(graph_data_list)
        torch.save((data, slices), self.processed_paths[0])


class TumorBrainDataset(Dataset):
    def __init__(self, root, label_path, transform=None, pre_transform=None):
        self.label_path = label_path
        self.are_directed = True  # if graphs in dataset directed
        # TODO: add check if processed
        self.label_encoder = preprocessing.LabelEncoder()
        self.labels = init_graph_labels(label_path, self.label_encoder)

        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "graphml")

    @property
    def raw_file_names(self):
        return Path(self.raw_dir).iterdir()

    @property
    def processed_file_names(self) -> List[str]:
        if Path(self.processed_dir).exists():
            return sorted(Path(self.processed_dir).glob("data_*.pt"))

        return []

    @property
    def num_classes(self):
        # TODO: just for Brain dataset for now!
        return 7

    def process(self):
        skipped = []
        idx = 0
        for i, graph_path in enumerate(self.raw_file_names):
            print(f"{i}: processing {graph_path.resolve()}")

            graph_name = graph_path.stem.split("_")[1]  # remove meanSum prefix
            y = get_label_for_graph(self.labels, graph_name)

            if y is None:
                print(f"Skip graph {graph_name} as it has no label")
                skipped.append(graph_name)
                continue

            iG: ig.Graph = ig.load(graph_path)
            # is_directed = iG.is_directed()
            edge_index, edges_weights = get_edges_with_weghts_from_igraph(iG)
            x = get_degree_matrix(iG)
            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edges_weights)
            print(f"{idx}| Saving Data object {data}")
            torch.save(data, os.path.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1
        print(f"Skipped {len(skipped)}: {skipped}")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        return data


# https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.get_laplacian
