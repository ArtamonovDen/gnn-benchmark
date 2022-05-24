

from dataset.graphml_dataset import GraphmlInMemoryDataset
from torch_geometric.datasets import TUDataset
from dataset.transforms import TransformController

from dataset.tu_dataset import TUDatasetWrapper


class DatasetController:

    @classmethod
    def get_transform(cls, type, transform_type, cat):
        max_degree, max_diam = 0, 0

        if type in GraphmlInMemoryDataset.Type.ALL:
            max_degree = GraphmlInMemoryDataset.get_max_degree(type)
            max_diam = GraphmlInMemoryDataset.get_max_diameter(type)

        elif TUDatasetWrapper.is_supported(type):
            max_degree = TUDatasetWrapper.get_max_degree(type)
            max_diam = TUDatasetWrapper.get_max_diameter(type)

        else:
            raise ValueError(f"Datset of type {type} is not supported")

        transform = TransformController.get_transform(
            transform_type,
            max_degree=max_degree,
            max_diam=max_diam,
            cat=cat,
        )
        return transform

    @classmethod
    def get_dataset(cls, type, root, transform_type, cat=True, label_path=None):

        pre_transform = cls.get_transform(type, transform_type, cat)

        if type in GraphmlInMemoryDataset.Type.ALL:
            return GraphmlInMemoryDataset(root=root, type=type, pre_transform=pre_transform, label_path=label_path)

        if TUDatasetWrapper.is_supported(type):
            return TUDataset(
                root=root, name=type, use_node_attr=True, use_edge_attr=True, pre_transform=pre_transform,
            )
        raise ValueError(f"Datset of type {type} is not supported")
