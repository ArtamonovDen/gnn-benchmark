from cmath import log
import json
import logging
import os
import sys
import torch
from tqdm import tqdm
from datetime import datetime
from dataset.dataset_controller import DatasetController
from train_args import parse
from dataset.graphml_dataset import GraphmlInMemoryDataset
from torch.optim.lr_scheduler import StepLR


from models.model import ModelController


from torch_geometric.loader import DataLoader
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


def get_dataset(root, type):
    # root = "/home/friday/projects/hse_gnn/datasets/cbs-datasets/BrainfMRI"
    # TODO: do smt with type

    # TODO: add a way to use TUDatasets

    ds = GraphmlInMemoryDataset(root=root, type=type)
    return ds





"""
Customize:
    * dataset and type
    * model
    * device?
    * hiddem layer size?

Add:
    * wnb?
    * validation
    * best model chosing
"""


def upgrade_model_state(cur_metric, best_metric, metric_type, model_path):
    if cur_metric > best_metric:
        logging.info("Update best model by %s from %f:.4 to %f:.4f", "acc", metric_type, best_metric, cur_metric)
        torch.save(model.state_dict(), model_path)
        return cur_metric
    return best_metric


def make_val_split(dataset, size=0.2):
    """
    Split graph dataset into train/test
    """
    val_bound = int(len(dataset) * size)
    train_dataset, test_dataset = dataset[:val_bound], dataset[:val_bound]
    return train_dataset, test_dataset


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,  format='[%(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %I:%M:%S'
    )

    args = parse()

    logging.info("Running train script with configuration: \n %s", args)

    torch.manual_seed(42)
    batch_size = args.batch_size

    dataset = DatasetController.get_dataset(args.type, args.root)
    dataset = dataset.shuffle()

    train_dataset, test_dataset = make_val_split(dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelController.get_model(
        args.model,
        num_node_features=dataset.num_node_features,
        hidden_channels=args.hidden,
        num_classes=dataset.num_classes,
    )(args.model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # TODO configurable lr
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_num = args.epoch

    best_val_acc, best_val_f1 = 0, 0
    model_path = os.path.join(args.save, f"{datetime.now().isoformat()}", f"model_{args.model}")
    # TODO: add file with cofiguration
    best_acc_model_path = os.path.join(model_path, "best_acc.pth")
    best_f1_model_path = os.path.join(model_path, "best_f1.pth")


    for epoch in range(epoch_num):
        logging.info(f"Epoch: {epoch:03d}. Start training")
        loss = train()

        logging.info(f"Epoch: {epoch:03d}. Start validation")
        train_acc = test(train_loader)
        scheduler.step()
        with torch.no_grad():
            test_acc, test_f1 = test(test_loader)

        logging.info(
            f"Epoch: {epoch:03d},  Loss: {loss:.4f},",
            f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}",
        )

        best_val_acc = upgrade_model_state(test_acc, best_val_acc, "accuracy", best_acc_model_path)
        best_val_f1 = upgrade_model_state(test_f1, best_val_f1, "f1", best_f1_model_path)

