import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-num_nodes", type=int, default=500, help="Number of nodes to take in training")
    parser.add_argument("-num_clusters", type=int, default=50, help="Number of weight-sharing clusters to form")
    parser.add_argument("-num_iters", type=int, default=30, help="Number of iterations to train the HMLN")
    parser.add_argument("-dataset", type=str, default="cora", choices=["cora", "citeseer", "pubmed"],
                        help="Choose the dataset name")
    parser.add_argument("-spec_model", type=str, default="gcn", choices=["gcn", "gs", "gat"],
                        help="Select the specification DNN.")
    parser.add_argument("-nuv_model", type=str, default="gat", choices=["gcn", "gs", "gat"],
                        help="Select the NUV DNN.")

    opt = parser.parse_args()



if __name__ == "__main__":
    main()