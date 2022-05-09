import os
import torch
from tqdm import tqdm
from dataset.graphml_dataset import GraphmlInMemoryDataset
from models.graph_gcn import GraphGCN
from models.graph_gin import GIN
from models.model import ModelController


from models.vanila_gcn import VanilaGCN
from dataset.brain import TumorMetDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score


def train():
    model.train()
    loss_all = 0.0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_attr,
            batch=data.batch,
        )
        loss = criterion(out, data.y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_loader)  # TODO: loss over batch?


def test(loader):
    model.eval()

    correct = 0
    total = 0
    for data in tqdm(loader):
        data = data.to(device)
        outputs = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_attr,
            batch=data.batch,
        )

        _, pred = torch.max(outputs, 1)
        correct += int((pred == data.y).sum())
        total += data.y.size(0)  # TODO: but seems we do not need it?

    acc = correct / total
    f1 = f1_score(data.y.cpu(), pred.cpu())  # TODO: check if it works

    return acc, f1


def get_dataset():
    root = "/home/friday/projects/hse_gnn/datasets/cbs-datasets/BrainfMRI"
    label_path = os.path.join(root, "labels.txt")
    ds = GraphmlInMemoryDataset(
        root=root, label_path=label_path, type=GraphmlInMemoryDataset.Type.BRAIN
    )
    return ds


def get_model(model_name):

    model = ModelController.get_model(
        model_name,
        num_node_features=dataset.num_node_features,
        hidden_channels=256,
        num_classes=dataset.num_classes,
    )

    return model


"""
Customize:
    * dataset and type
    * model
    * train/test ratio
    * batch_size
    * device?
    * lr + use dynamic lr?
    * ecpoch num
    * hiddem layer size?

Add:
    * wnb?
    * validation
    * best model chosing
"""

if __name__ == "__main__":

    torch.manual_seed(42)

    dataset = get_dataset()
    dataset = dataset.shuffle()
    batch_size = 8

    train_dataset = dataset[:100]
    test_dataset = dataset[100:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # TODO configurable lr
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 171):
        print(f"Epoch: {epoch:03d}. Start training")
        loss = train()
        print(f"Epoch: {epoch:03d}. Start validation")
        train_acc = test(train_loader)
        with torch.no_grad():  # TODO: really need?
            test_acc, test_f1 = test(test_loader)
        print(
            f"Epoch: {epoch:03d},  Loss: {loss:.4f},",
            f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}",
        )

        # TODO: check and save if needed
        # PATH = "./cifar_net.pth"
        # torch.save(model.state_dict(), PATH)
