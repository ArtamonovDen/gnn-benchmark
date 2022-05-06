import os

from dataset.graphml_dataset import GraphmlInMemoryDataset

# from datasets.brain_large import BrainDataset

root = "/home/friday/projects/hse_gnn/datasets/cbs-datasets/BrainfMRI"
label_path = os.path.join(root, "labels.txt")
ds = GraphmlInMemoryDataset(
    root=root, label_path=label_path, type=GraphmlInMemoryDataset.Type.BRAIN
)
print(ds)
