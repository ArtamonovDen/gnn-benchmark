

from dataset.graphml_dataset import GraphmlInMemoryDataset
from torch_geometric.datasets import TUDataset
from dataset.transforms import TransfromController

from dataset.tu_dataset import TUDatasetWrapper


class DatasetController:
    @classmethod
    def get_dataset(cls, type, root, transform_type):
        if type in GraphmlInMemoryDataset.Type.ALL:
            pre_transform = TransfromController.get_transform(
                transform_type,
                max_degree=GraphmlInMemoryDataset.max_degree2dataset,
                max_diam=GraphmlInMemoryDataset.max_diam2dataset,
            )
            return GraphmlInMemoryDataset(root=root, type=type, pre_transform=pre_transform)

        if TUDatasetWrapper.is_supported(type):
            pre_transform = TransfromController.get_transform(
                transform_type,
                max_degree=TUDatasetWrapper.get_max_degree(type),
                max_diam=TUDatasetWrapper.get_max_diameter(type)
            )
            return TUDataset(
                root=root, name=type, use_node_attr=True, use_edge_attr=True, pre_transform=pre_transform
            )
        raise ValueError(f"Datset of type {type} is not supported")
