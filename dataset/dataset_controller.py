

from dataset.graphml_dataset import GraphmlInMemoryDataset
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T

from dataset.tu_dataset import TUDatasetWrapper


class DatasetController:
    @classmethod
    def get_dataset(cls, type, root, **kwargs):
        if type in GraphmlInMemoryDataset.Type.ALL:
            return GraphmlInMemoryDataset(root=root, type=type, **kwargs)

        if TUDatasetWrapper.is_supported(type):
            max_degree = TUDatasetWrapper.get_max_degree(type)
            return TUDataset(
                root=root, name=type, use_node_attr=True, use_edge_attr=True, transform=T.OneHotDegree(max_degree=max_degree)
            )
        raise ValueError(f"Datset of type {type} is not supported")

