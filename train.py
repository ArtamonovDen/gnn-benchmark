import os
import torch
from tqdm import tqdm
from dataset.graphml_dataset import GraphmlInMemoryDataset
from models.graph_gcn import GraphGCN
from models.graph_gin import GIN


from models.vanila_gcn import VanilaGCN
from dataset.brain import TumorMetDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F


def train():
    model.train()
    loss_all = 0.0
    for data in tqdm(train_loader):  # Iterate in batches over the training dataset.
        optimizer.zero_grad()
        data = data.to(device)
        out = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_attr,
            batch=data.batch,
        )
        # loss = F.nll_loss(out, data.y)
        loss = criterion(out, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()  # Update parameters based on gradients.
    return loss_all / len(train_loader.dataset)  # TODO: same as len(train_loader)?


def test(loader):
    model.eval()

    correct = 0
    for data in tqdm(loader):  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_attr,
            batch=data.batch,
        )

        # pred = out.max(dim=1)[1]
        # correct += pred.eq(data.y).sum().item()

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def get_dataset():
    root = "/home/friday/projects/hse_gnn/datasets/cbs-datasets/BrainfMRI"
    label_path = os.path.join(root, "labels.txt")
    ds = GraphmlInMemoryDataset(
        root=root, label_path=label_path, type=GraphmlInMemoryDataset.Type.BRAIN
    )
    return ds


def get_model():
    # model = VanilaGCN(
    #     num_node_features=dataset.num_node_features,
    #     hidden_channels=256,
    #     num_classes=dataset.num_classes,
    # )
    model = GIN(
        num_node_features=dataset.num_node_features,
        hidden_channels=256,
        num_classes=dataset.num_classes,
    )
    return model


if __name__ == "__main__":

    # torch.manual_seed(42)

    dataset = get_dataset()
    dataset = dataset.shuffle()
    batch_size = 8

    train_dataset = dataset[:100]
    test_dataset = dataset[100:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 171):
        print(f"Epoch: {epoch:03d}. Start training")
        loss = train()
        print(f"Epoch: {epoch:03d}. Start validation")
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(
            f"Epoch: {epoch:03d},  Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )
