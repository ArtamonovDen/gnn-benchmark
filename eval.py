
import argparse
import os
import logging
import json
import numpy as np
from pathlib import Path
import sys


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,  format='[%(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %I:%M:%S'
    )

    parser = argparse.ArgumentParser(description="GCN models train script")
    parser.add_argument("-r", "--root",  type=str, help="Root directory to search files with metrics")
    parser.add_argument("-f", "--file", type=str, help="Name of file with metrics to search")
    parser.add_argument("-s", "--save", type=str, help="Path to save results")


    args = parser.parse_args()

    acc_list, f1_list = [], []
    
    #  recursive globbing over sub directories of root
    for metric_file in sorted(Path(args.root).glob(f"**/{args.file}.py")):
        with open(metric_file, "r") as f:
            logging.info("Process %s", str(metric_file))
            metrics = json.load(f)
            acc_list.append(float(metrics["acc"]))
            f1_list.append(float(metrics["f1"]))

    accs, f1s = np.asarray(acc_list), np.asarray(f1_list)

    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_f1, std_f1 = np.mean(f1s), np.std(f1s)

    logging.info("Mean accuracy is %.4f, std accuracy %.4f", mean_acc, std_acc)
    logging.info("Mean f1 is %.4f, std f1 %.4f", mean_f1, std_f1)

    with open(os.path.join(args.save, "test_metric_results.json"), "w") as f:
        json_results = json.dumps({
            "root": args.root,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
        }, indent=4, sort_keys=True)
        f.write(json_results)