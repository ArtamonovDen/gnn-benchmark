import json
import logging
import os
import sys
import torch
import wandb
from tqdm import tqdm
from datetime import datetime
from dataset.dataset_controller import DatasetController
from train_args import parse
from torch.optim.lr_scheduler import StepLR

from models.model_controller import ModelController

from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score


def train():
    model.train()
    loss_all = 0.0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        # TODO: distingush weight and edge attr
        edge_weight = None
        out = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            edge_weight=edge_weight,
            batch=data.batch,
        )
        loss = criterion(out, data.y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_loader)


def test(loader, binary=True):
    model.eval()

    f1_average = "binary" if binary else "weighted"

    correct = 0
    total = 0
    total_y, total_pred = None, None
    edge_weight = None # TODO
    for data in tqdm(loader):
        data = data.to(device)
        outputs = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            edge_weight=edge_weight,
            batch=data.batch,
        )

        _, pred = torch.max(outputs, 1)
        correct += int((pred == data.y).sum())
        total += data.y.size(0)

        # store all labels to get metrics
        if total_y is None and total_pred is None:
            total_y = data.y.cpu()
            total_pred = pred.cpu()
        else:
            total_y = torch.cat((total_y, data.y.cpu()), dim=0)
            total_pred = torch.cat((total_pred, pred.cpu()), dim=0)

    acc = correct / total
    f1 = f1_score(total_y, total_pred, average=f1_average) # TODO: choose average. different for binaries?

    return acc, f1


def choose_device(args):
    return torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def upgrade_model_state(cur_metric, best_metric, metric_type, model_path):
    if cur_metric > best_metric:
        logging.info("Update best model by %s from %.4f to %.4f", metric_type, best_metric, cur_metric)
        torch.save(model.state_dict(), model_path)
        return cur_metric
    return best_metric


def make_val_split(dataset, size=0.5):
    """
    Split graph dataset into train/test
    """
    val_bound = int(len(dataset) * size)
    train_dataset, test_dataset = dataset[val_bound:], dataset[:val_bound]
    logging.info("Split dataset by ratio %f. Train dataset size %d, test dataset size %d", size, len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,  format='[%(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %I:%M:%S'
    )

    args = parse()

    model_path = os.path.join(args.save, f"dataset_{args.type}", f"{datetime.now().isoformat()}", f"model_{args.model}")
    os.makedirs(model_path, exist_ok=True)

    # config = vars(args)


    logging.info("Running train script with configuration: \n %s", args)

    torch.manual_seed(42)
    dataset = DatasetController.get_dataset(args.type, args.root)
    dataset = dataset.shuffle()
    train_dataset, test_dataset = make_val_split(dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    is_binary = dataset.num_classes == 2

    device = choose_device(args)

    model = ModelController.get_model(
        model_name=args.model,
        num_node_features=dataset.num_node_features,
        hidden_channels=args.hidden_layers,
        num_classes=dataset.num_classes,
    ).to(device)

    logging.info("Running training procedure of model %s and dataset %s\n Model config: %s\n Argumetns %s", args.model, args.type, model, vars(args))

    with open(os.path.join(model_path, "config.json"), "w") as f:
        config = vars(args)
        config["model_config"] = str(model)
        json_config = json.dumps(config, indent=4, sort_keys=True)
        f.write(json_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # TODO configurable lr
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc, best_val_f1 = 0, 0
    best_acc_model_path = os.path.join(model_path, "best_acc.pth")
    best_f1_model_path = os.path.join(model_path, "best_f1.pth")

    with wandb.init(project=args.wandb_project):
        wandb.watch(model)
        wandb.config.update(args)
        wandb.config.update({"model_path": model_path})

        for epoch in range(args.epoch):
            logging.info("Epoch: %03d. Start training", epoch)
            wandb.log({'epoch': epoch})
            loss = train()

            logging.info("Epoch: %03d. Start validation", epoch)
            with torch.no_grad():
                train_acc, train_f1 = test(train_loader, binary=is_binary)
                test_acc, test_f1 = test(test_loader, binary=is_binary)

            best_val_acc = upgrade_model_state(test_acc, best_val_acc, "accuracy", best_acc_model_path)
            best_val_f1 = upgrade_model_state(test_f1, best_val_f1, "f1", best_f1_model_path)

            scheduler.step()

            logging.info(
                "Epoch: %03d,  Loss: %.4f, Train Acc: %.4f, Train F1: %.4f, Test Acc: %.4f, Test F1: %.4f",
                epoch, loss, train_acc, train_f1, test_acc, test_f1
            )
            wandb.log({"train_acc": train_acc, "test_acc": test_acc, "train_f1": train_f1, "test_f1": test_f1, "loss": loss})

    with open(os.path.join(model_path, "val_metric.json"), "w") as f:
        json_config = json.dumps({
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1,
            "last_train_acc": train_acc,
            "last_train_f1": train_f1,
            "loss": loss,
        }, indent=4, sort_keys=True)
        f.write(json_config)
