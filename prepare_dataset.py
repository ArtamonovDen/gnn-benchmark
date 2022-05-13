import os
from dataset.dataset_controller import DatasetController
from dataset.graphml_dataset import GraphmlInMemoryDataset
from torch_geometric import utils


def check_max_degree(dataset):
    max_out_dergree = max(utils.degree(data.edge_index[1], data.num_nodes).max() for data in dataset)
    max_in_degree = max(utils.degree(data.edge_index[0], data.num_nodes).max() for data in dataset)
    max_degree = max(max_in_degree, max_out_dergree)
    print(f"Max degree {max_degree.item()}")


if __name__ == "__main__":
    root = "/home/friday/projects/hse_gnn/datasets/cbs-datasets/BrainfMRI"
    label_path = os.path.join(root, "labels.txt")

    dataset = DatasetController.get_dataset(type=GraphmlInMemoryDataset.Type.BRAIN, root=root, label_path=label_path)
    print(dataset)
