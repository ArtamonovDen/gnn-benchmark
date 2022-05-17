import argparse

from models.model_controller import ModelController


def parse():
    parser = argparse.ArgumentParser(description="GCN models train script")
    parser.add_argument("-m", "--model",  required=True, type=str, help="Model name to train", choices=ModelController.get_available_models())
    parser.add_argument("-r", "--root", required=True, type=str, help="Root path for dataset")
    parser.add_argument("-t", "--type", required=True, type=str, help="GraphML dataset type")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="Epoch number")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--cpu", action='store_true', help="Run train on CPU")
    parser.add_argument("--weighted", action='store_true', help="Graph contains edges weights")
    parser.add_argument("-y", "--hidden_layers", type=int, help="List of hidden layers dimensions", nargs="+")
    parser.add_argument("-s", "--save", type=str, default="./model_snapshots/", help="Path to save model to")
    parser.add_argument("-p", "--wandb_project", type=str, help="Name of the WandB project")
    parser.add_argument("-v", "--val_ratio", type=float, default=0.3, help="Train/Validation split ratio")
    parser.add_argument("--conv_num", type=int, default=3, help="Number of convolution layers in GCN")
    parser.add_argument("--conv_pooling", type=str, default="mean", help="Pooling strategy to get whole graph embedding", choices=["max","mean","add"])

    args = parser.parse_args()
    return args
