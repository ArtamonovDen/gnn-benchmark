import os

from dataset.brain import TumorBrainInMemoryDataset, TumorBrainDataset

# from datasets.brain_large import BrainDataset

root = "/home/friday/projects/hse_gnn/datasets/TumorMet/Brain"
label_path = os.path.join(root, "sample_sheet.tsv")
ds = TumorBrainDataset(root, label_path)
print(ds)
