import argparse

featurefile = "feat.txt"
adjfile = "A.txt"
labelfile = "label.txt"
trainmaskfile = "train_mask.txt"
validmaskfile = "val_mask.txt"
testmaskfile = "test_mask.txt"

GRDATA = "grexp/"
BMFDATA = "BMF/"
BPDATA = "BPProbs/"
RANKEXPDATA = "rankexp/"


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





if __name__ == "__main__":
    main()