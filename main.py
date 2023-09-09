import argparse

import pandas as pd
import torch

from src.explanation import *
from src.LRInfer import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-start_rank", type=int, default=300, help="Starting value of the low rank approximation")
    parser.add_argument("-end_rank", type=int, default=500, help="Ending value of the low rank approximation")
    parser.add_argument("-inc_rank", type=int, default=50, help="Increment value for the low rank approximation")
    parser.add_argument("-exp_limit_nodes", type=int, default=1, help="Limiting value for number of explanation nodes")
    parser.add_argument("-num_classes", type=int, default=5, help="Number of classes for the dataset")
    parser.add_argument("-dataset_name", type=str, default="bashapes", choices=["bashapes", "bacommunity", "treecycle", "treegrid"],
                         help="Name of the dataset")

    opt = parser.parse_args()

    feat = torch.tensor(pd.read_csv(f"{opt.dataset_name}/{featurefile}", header=None, sep='\t').to_numpy().astype(np.float32))
    label = torch.tensor(pd.read_csv(f"{opt.dataset_name}/{labelfile}", header=None, sep='\t').to_numpy().flatten().astype(int))
    _A = pd.read_csv(f"{opt.dataset_name}/{adjfile}", header=None, sep='\t').to_numpy()

    train_mask = None
    valid_mask = None
    test_mask = None

    if trainmaskfile:
        train_mask = torch.tensor(
            pd.read_csv(f"{opt.dataset_name}/{trainmaskfile}", header=None, sep='\t').to_numpy()[:, 0].flatten().astype(bool))
    if validmaskfile:
        valid_mask = torch.tensor(
            pd.read_csv(f"{opt.dataset_name}/{validmaskfile}", header=None, sep='\t').to_numpy()[:, 0].flatten().astype(bool))
    if testmaskfile:
        test_mask = torch.tensor(
            pd.read_csv(f"{opt.dataset_name}/{testmaskfile}", header=None, sep='\t').to_numpy()[:, 0].flatten().astype(bool))

    org_graph = Graph(opt.dataset_name, feat, label, _A, train_mask, valid_mask, test_mask)
    G = org_graph.generate()
    print(f"Original Graph: {G}")

    print(f"Generating explanations for original graph G ... ")
    generate_grexpdata(G, arg=opt)
    print(f"Generating explanations for low ranked graph G' ... ")
    print(f"Starting rank: {opt.start_rank} Stop rank at: {opt.end_rank} Increase rank by: {opt.inc_rank}")
    generate_rankexpdata(G, arg=opt)
    belief_propagation(arg = opt)
    get_results(G, arg=opt)


if __name__ == "__main__":
    main()