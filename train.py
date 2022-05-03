import os
import torch
from tqdm import tqdm


from models.vanila_gcn import VanilaGCN
from dataset.brain import TumorBrainDataset
from torch_geometric.loader import DataLoader


def train():
    model.train()

    for data in tqdm(train_loader):  # Iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.edge_attr, data.batch
        )  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in tqdm(loader):  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def get_dataset():
    root = "/home/friday/projects/hse_gnn/datasets/TumorMet/Brain"
    label_path = os.path.join(root, "sample_sheet.tsv")
    ds = TumorBrainDataset(root, label_path)
    return ds


if __name__ == "__main__":

    # torch.manual_seed(42)

    dataset = get_dataset()
    dataset = dataset.shuffle()
    batch_size = 16

    train_dataset = dataset[150:]
    test_dataset = dataset[:150]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = VanilaGCN(
        num_node_features=dataset.num_node_features,
        hidden_channels=256,
        num_classes=dataset.num_classes,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 171):
        print(f"Epoch: {epoch:03d}. Start training")
        train()
        print(f"Epoch: {epoch:03d}. Start validation")
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )
