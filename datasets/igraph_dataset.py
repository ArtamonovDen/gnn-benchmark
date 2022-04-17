from pathlib import Path
import pandas as pd
import os
import torch
from sklearn import preprocessing
import igraph as ig
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import InMemoryDataset, Data

from typing import Dict, Union, List, Optional, Tuple, Callable


class IGraphDataset(InMemoryDataset):

    IGRAPH_SUPPORTED_FORAMATS = "graphml"

    def __init__(
        self,
        root,
        label_path,
        transform=None,
        pre_transform=None,
    ):
        self.label_path = label_path

        self.label_encoder = preprocessing.LabelEncoder()
        self.labels = None
        self.init_graph_labels()

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def edge_attributes(self):
        """
        Should be overridden
        """
        return []

    @property
    def is_undirected():
        return True

    @property
    def node_attributes(self):
        """
        Should be overridden
        """
        return []

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "graphml")

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["graph_data.pt"]

    @property
    def raw_file_names(self):
        return sorted(Path(self.raw_dir).iterdir())

    def init_graph_labels(self):
        # TODO: add a way to override
        labels_df = pd.read_csv(self.label_path, delimiter="\t")
        encoded_labels = self.label_encoder.fit_transform(labels_df["labels"])
        self.labels = torch.LongTensor(encoded_labels)

    def get_edges_from_igraph(self, iG: ig.Graph) -> torch.LongTensor:
        print("Obtaining edge matrix")
        e_num = iG.ecount()
        coo_format_edges = list(zip(*iG.get_edgelist()))

        # add back-edges manually for undirected graph
        source_nodes, sink_nodes = coo_format_edges[0], coo_format_edges[1]
        coo_format_edges[0] += sink_nodes
        coo_format_edges[1] += source_nodes
        pyg_edge_index = torch.LongTensor(coo_format_edges)
        print(f"Edge coo matrix has shape {pyg_edge_index.shape}")

        # TODO: SEEMS WE NEED TO ADD BACK EDGE MANUALLY FOR UNDIRECTED GRAPHS!

        assert pyg_edge_index.shape == (2, e_num * 2)
        return pyg_edge_index

    def get_raw_edge_arrts_from_igraph(
        self, iG: ig.Graph
    ) -> Optional[torch.FloatTensor]:
        print("Obtaining edge attibutes...")
        if not self.edge_attributes:
            return None
        e_num = iG.ecount()
        edge_attrs = []
        for edge in iG.es:
            one_edge_attrs = [edge[attr_name] for attr_name in self.edge_attributes]
            edge_attrs.append(one_edge_attrs)
        # TODO add feature converter if not float?
        pyg_edge_attrs = torch.FloatTensor(edge_attrs)
        print(f"Edge attribute matrix has shape {pyg_edge_attrs.shape}")

        assert pyg_edge_attrs.shape == (e_num, len(self.edge_attributes))
        return pyg_edge_attrs

    def get_raw_node_arrts_from_igraph(
        self, iG: ig.Graph
    ) -> Optional[torch.FloatTensor]:
        print("Obtaining node attibutes...")
        if not self.node_attributes:
            return None
        n_num = iG.vcount()
        node_attrs = []
        for node in iG.vs:
            one_node_attrs = [node[attr_name] for attr_name in self.node_attributes]
            node_attrs.append(one_node_attrs)

            # TODO: encode one hot <- сначала enc.fit() от всех категорий всех графов

        # TODO add feature converter if not float?
        pyg_node_attrs = torch.FloatTensor(node_attrs)
        print(f"Edge attribute matrix has shape {pyg_node_attrs.shape}")
        assert pyg_node_attrs.shape == (n_num, len(self.node_attributes))
        return pyg_node_attrs

    def convert_raw_graphs(self) -> List[Data]:
        graph_data_list = []
        for i, graph_path in enumerate(
            self.raw_file_names
        ):  # TODO: must be sure in labels order
            print(f"Processing {graph_path.resolve()}")
            iG: ig.Graph = ig.load(graph_path)

            pyg_edge_index = self.get_edges_from_igraph(iG)
            # TODO: broken for now: self.get_raw_edge_arrts_from_igraph(iG)
            pyg_edge_attrs = None
            pyg_x = self.get_raw_node_arrts_from_igraph(iG)

            pyg_y = self.labels[i]

            # Create Data object
            graph = Data(
                x=pyg_x, edge_index=pyg_edge_index, edge_attr=pyg_edge_attrs, y=pyg_y
            )
            graph_data_list.append(graph)

        data, slices = self.collate(graph_data_list)
        return data, slices
