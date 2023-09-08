import pandas as pd
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import dgl
import networkx as nx
import scipy.sparse as sp

import random
import math
import pickle as pk



class Graph:
    '''
    Class: Graph
    '''
    def __init__(self, dataset_name, feat, label, A, train_mask=None, valid_mask=None, test_mask=None):
        self.dataset_name = dataset_name
        self.feat = feat
        self.label = label
        self.A = A

        self.train_mask = train_mask
        self.valid_mask = valid_mask
        self.test_mask = test_mask


    def generate(self):
        '''
        This function generates a dgl graph with inputs: features, labels and adjacency matrix.
        It also separates train and test data with masks.
        '''
        adjacency_sparse = sp.coo_matrix(torch.tensor(self.A))
        G = dgl.from_scipy(adjacency_sparse)
        G.ndata['feat'] = self.feat
        G.ndata['label'] = self.label
        num_nodes = G.num_nodes()

        if self.train_mask is not None and self.valid_mask is not None and self.test_mask is not None:
          G.ndata['train_mask'] = self.train_mask
          G.ndata['val_mask'] = self.valid_mask
          G.ndata['test_mask'] = self.test_mask
        else:
          train_mask_indices = sorted(random.sample(range(num_nodes), math.ceil(num_nodes*0.8)))
          val_mask_indices = sorted(random.sample([i for i in range(num_nodes) if i not in train_mask_indices], math.ceil(num_nodes*0.15)))
          test_mask_indices = [i for i in range(num_nodes) if i not in train_mask_indices and i not in  val_mask_indices]

          train_mask = torch.tensor([True if i in train_mask_indices else False for i in range(num_nodes)])
          val_mask = torch.tensor([True if i in val_mask_indices else False for i in range(num_nodes)])
          test_mask = torch.tensor([True if i in test_mask_indices else False for i in range(num_nodes)])

          G.ndata['train_mask'] = train_mask
          G.ndata['val_mask'] = val_mask
          G.ndata['test_mask'] = test_mask
        return G

    def generate_noisy(self, ADD_PROB = 0.15, DEL_PROB = 0.15, FEAT_NOISE_INTENSITY = 0.15):
        '''
        This function generates a dgl graph with noise induced with add, delete and feature noise probability.
        '''
        mu=0
        sigma = 0.05
        # correct_dir(dataset_name, gnn)
        distance_matrix = pk.load(open(f"distance_matrix.pkl", "rb"))

        G = self.generate()
        closest_nodes = self.find_closest_nodes(distance_matrix, G)
        farthest_nodes = self.find_farthest_nodes(distance_matrix, G)

        # Add feature noise
        feat_noise = torch.from_numpy(np.random.normal(mu, sigma, size=G.ndata['feat'].shape)).float().to(device)
        G.ndata["feat"] = G.ndata["feat"] + feat_noise * FEAT_NOISE_INTENSITY

        del_indices = []
        del_edges = []
        added_edges = []

        # Remove noisy edges
        for idx, (u, v) in enumerate(zip(G.edges()[0], G.edges()[1])):
            edge = (u.item(), v.item())
            reverse_edge = (v.item(), u.item())
            if edge in closest_nodes or reverse_edge in closest_nodes:
                if random.random() < DEL_PROB:
                    del_indices.append(idx)
                    del_edges.append(edge)
                    del_edges.append(reverse_edge)
        G.remove_edges(torch.tensor(del_indices, dtype=torch.int64).to(device))
        print("Total Edges Removed = ", len(del_edges))

        # Add noisy edges
        for edge in farthest_nodes:
            if random.random() < ADD_PROB:
                G.add_edges(edge[0], edge[1])
                G.add_edges(edge[1], edge[0])
                added_edges.append((edge[0], edge[1]))
                added_edges.append((edge[1], edge[0]))
        print("Total Edges added = ", len(added_edges))

        return G

    def find_closest_nodes(self, dist_matrix_np, graph):
        '''
          Find closest connections/edges to remove them later
        '''
        nx_graph = graph.cpu().to_networkx()
        adj_matrix = nx.adjacency_matrix(nx_graph)
        adj_matrix = adj_matrix.toarray()

        truncated_dist_matrix = np.multiply(adj_matrix, dist_matrix_np)
        node_pairs = np.where((truncated_dist_matrix < 0.05) & (truncated_dist_matrix > 0))
        edge_list = []

        for i,j in zip(node_pairs[0], node_pairs[1]):
            edge = (i, j)
            reverse_edge = (j, i)
            if i != j and edge not in edge_list and reverse_edge not in edge_list and j in graph.successors(i):
                if len(graph.successors(i)) > 1:
                    edge_list.append(edge)

        return edge_list


    def find_farthest_nodes(self, dist_matrix_np, graph):
        '''
          Find farthest nodes which has no connections/edges to add them later
        '''
        node_pairs = np.where(dist_matrix_np > (dist_matrix_np.max()-0.25))
        edge_list = []

        for i, j in zip(node_pairs[0], node_pairs[1]):
            edge = (i, j)
            reverse_edge = (j, i)
            if edge not in edge_list and reverse_edge not in edge_list and j not in graph.successors(i):
                edge_list.append((i, j))

        return edge_list