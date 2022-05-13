

from dataset.graphml_dataset import GraphmlInMemoryDataset


class DatasetController:
    @classmethod
    def get_dataset(cls, type, root):
        if type in GraphmlInMemoryDataset.Type.ALL:
            return GraphmlInMemoryDataset(root=root, type=type)
        # TODO: return TUDataset

