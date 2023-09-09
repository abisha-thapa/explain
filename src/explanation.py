import pandas as pd
import os
import dgl
from dgl.nn import GNNExplainer

from train import train
from src import GCN, Graph
from utils import *

def generate_grexpdata(G, arg = None):
    '''

    :param G:
    :param arg:
    :return:
    '''
    if not os.path.exists(f"data/{arg.dataset_name}/{GRDATA}"):
        os.mkdir(f"data/{arg.dataset_name}/{GRDATA}")
    model, preds = train(GCN, G)
    explainer = GNNExplainer(model, num_hops=2, log=False)
    for idx in range(G.num_nodes()):
        # if label[idx] ==0:
        nc, sg, fm, em = explainer.explain_node(node_id = idx, graph=G, feat = G.ndata['feat'])
        if len(sg.nodes()) > arg.exp_limit_nodes:
            nodes = sg.ndata[dgl.NID].tolist()
            ofile = open(f"data/{arg.dataset_name}/{GRDATA}{id}.txt",'w')
            src, dst = sg.edges()
            for s, d, em in zip(src.numpy(), dst.numpy(), em.numpy()):
                s1 = str(nodes[s])
                d1 = str(nodes[d])
                ofile.write("-1"+","+s1+","+d1+","+str(em)+"\n")
            ofile.close()


def generate_rankexpdata(G, arg = None):
    '''

    :param G:
    :param arg:
    :return:
    '''
    if not os.path.exists(f"data/{arg.dataset_name}/{RANKEXPDATA}"):
        os.mkdir(f"data/{arg.dataset_name}/{RANKEXPDATA}")
    for r in range(arg.start_rank, arg.end_rank, arg.inc_rank):
        _D = pd.read_csv(f"data/{arg.dataset_name}/{BMFDATA}{r}.txt", header=None, sep=',').to_numpy()
        gr = Graph(arg.dataset_name, G.ndata['feat'], G.ndata['label'], _D, G.ndata['train_mask'], G.ndata['val_mask'], G.ndata['test_mask'])
        LG = gr.generate()
        model, lrpreds = train(GCN, LG)
        explainer = GNNExplainer(model, num_hops=2, log=False)
        for idx in range(G.num_nodes()):
            # if label[idx] ==0:
            nc, sg, fm, em = explainer.explain_node(node_id = idx, graph=LG, feat = LG.ndata['feat'])
            if len(sg.nodes()) > arg.exp_limit_nodes:
                nodes = sg.ndata[dgl.NID].tolist()
                ofile = open(f"data/{arg.dataset_name}/{RANKEXPDATA}{idx}.txt",'a')
                src, dst = sg.edges()
                for s, d, em in zip(src.numpy(), dst.numpy(), em.numpy()):
                    s1 = str(nodes[s])
                    d1 = str(nodes[d])
                    ofile.write(str(r)+","+s1+","+d1+","+str(em)+"\n")
                ofile.close()
        print("done "+str(r))

