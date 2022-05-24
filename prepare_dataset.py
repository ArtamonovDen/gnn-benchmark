import os
from dataset.dataset_controller import DatasetController
from dataset.graphml_dataset import GraphmlInMemoryDataset
from torch_geometric import utils
import logging
import sys


def check_max_degree(dataset):
    max_out_dergree = max(utils.degree(data.edge_index[1], data.num_nodes).max() for data in dataset)
    max_in_degree = max(utils.degree(data.edge_index[0], data.num_nodes).max() for data in dataset)
    max_degree = max(max_in_degree, max_out_dergree)
    print(f"Max degree {max_degree.item()}")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,  format='[%(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %I:%M:%S'
    )

    # root = "/home/friday/projects/hse_gnn/datasets/cbs-datasets/BrainfMRI"
    # type = GraphmlInMemoryDataset.Type.BRAIN
    root = "/home/friday/projects/hse_gnn/datasets/cbs-datasets/KidneyMetabolic"
    type = GraphmlInMemoryDataset.Type.KIDNEY
    label_path = os.path.join(root, "labels.txt")

    dataset = DatasetController.get_dataset(type=type, root=root, label_path=label_path, transform_type="deg") # TODO add transforms info
    print(dataset)
