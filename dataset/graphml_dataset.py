import logging
from pathlib import Path
import os
from typing import List
from sklearn import preprocessing
import torch
import igraph as ig
import pandas as pd


from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from dataset.igraph_tools import get_edges_with_weghts_from_igraph


class GraphmlInMemoryDataset(InMemoryDataset):
    class Type:
        BRAIN = "brain_fmri"
        KIDNEY = "kidney_metabolic"
        ALL = [BRAIN, KIDNEY]

    __degrees = {Type.BRAIN: 238, Type.KIDNEY: 105}
    __diameters = {Type.BRAIN: 2, Type.KIDNEY: 7}
    __classes = {Type.BRAIN: 2, Type.KIDNEY: 3}

    def __init__(self, type, root, transform=None, pre_transform=None, label_path=None):
        self.label_path = label_path
        self.type = type
        if label_path:
            # may be skipped if dataset is already preprocessed
            self.label_encoder = preprocessing.LabelEncoder()
            self.labels = self.init_graph_labels()

        super().__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @classmethod
    def get_max_degree(cls, type):
        return cls.__degrees[type]

    @classmethod
    def get_max_diameter(cls, type):
        return cls.__diameters[type]

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "graphml")

    @property
    def raw_file_names(self):
        if self.type == self.Type.BRAIN:
            return sorted(
                Path(self.raw_dir).iterdir(), key=lambda m: int(m.stem.split("_")[-1])
            )
        return sorted(Path(self.raw_dir).iterdir())

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.type}_graph_data.pt"]

    @property
    def classes2dataset(self):
        return {self.Type.BRAIN: 2, self.Type.KIDNEY: 3}

    @property
    def max_degree2dataset(self):
        return {self.Type.BRAIN: 238, self.Type.KIDNEY: 105}

    @property
    def max_diam2dataset(self):
        return {self.Type.BRAIN: 2, self.Type.KIDNEY: 7}

    @property
    def num_classes(self):
        return self.__classes.get(self.type)

    def init_graph_labels(self):

        lablels_mapping = pd.read_csv(self.label_path, sep=" ")
        lablels_mapping = lablels_mapping[["Samples", "labels"]]
        lablels_mapping = lablels_mapping.set_index("Samples")

        labels_cat = self.label_encoder.fit_transform(lablels_mapping["labels"].values)
        lablels_mapping["labels_cat"] = labels_cat
        labels = lablels_mapping["labels_cat"]
        return labels

    def get_label_for_graph(self, graph_name):
        if graph_name in self.labels:
            encoded_label = self.labels[graph_name]
            return torch.tensor(encoded_label, dtype=torch.long)
        return None

    def get_graph_name(self, graph_path: Path):
        if self.type == self.Type.BRAIN:
            return graph_path.stem.split("_")[1]
        return graph_path.stem

    def process(self):
        graph_data_list = []
        skipped = []
        for i, graph_path in enumerate(self.raw_file_names):
            print(f"{i}: processing {graph_path.resolve()}")

            graph_name = graph_path.stem
            y = self.get_label_for_graph(graph_name)
            if y is None:
                print(f"Skip graph {graph_name} as it has no label")
                skipped.append(graph_name)
                continue

            iG: ig.Graph = ig.load(graph_path)
            is_directed = iG.is_directed()
            edge_index, edge_attr = get_edges_with_weghts_from_igraph(iG)

            if not is_directed:
                edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr)
            data = Data(x=None, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data.num_nodes = len(iG.vs)
            graph_data_list.append(data)
            print(f"{i}| Saving Data object {data}")

        logging.info("Applying pre_transform function %r", self.pre_transform)
        if self.pre_transform is not None:
            graph_data_list = [self.pre_transform(data) for data in graph_data_list]

        logging.info("Skipped %d:%r", len(skipped), skipped)
        data, slices = self.collate(graph_data_list)
        torch.save((data, slices), self.processed_paths[0])
