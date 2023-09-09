import pandas as pd
import numpy as np

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math
import os

# dgl.backend.set_default_backend(default_dir='pytorch', backend_name='pytorch')

from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from statsmodels.stats.contingency_tables import mcnemar

from src import GCN, Graph
from train import train
from utils import *

def belief_propagation(arg = None):
    idnames = os.listdir(f"data/{arg.dataset_name}/{RANKEXPDATA}")
    for idx in idnames:
        getBPList(idx, arg)

def getBPList(idx, arg=None):
    datadir = f"data/{arg.dataset_name}/{RANKEXPDATA}"
    if idx.find(".txt")<0:
        return []
    if not os.path.exists(datadir+idx):
        return []
    if not os.path.exists(GRDATA+idx):
        return
    ifile = open(datadir+idx)
    edge_imp = {}
    counts = {}
    for ln in ifile:
        parts = ln.strip().split(",")
        if parts[1]==parts[2]:
            continue
        key = parts[1]+":"+parts[2]
        if key not in edge_imp:
            edge_imp[key]=float(parts[3])
            counts[key]=1
        else:
            edge_imp[key]=edge_imp[key] + float(parts[3])
            counts[key]=counts[key]+1
    ifile.close()
    for k in edge_imp.keys():
        v = edge_imp[k]/counts[k]
        edge_imp[k] = v
    # L = sorted(edge_imp.items(), key=lambda item: item[1],reverse=True)
    #print(L)

    FG = FactorGraph()

    for k in edge_imp.keys():
      parts = k.split(":")
      FG.add_nodes_from(parts)
      phi1 = DiscreteFactor(parts, [2, 2], [1,1,1,math.exp(edge_imp[k])])
      FG.add_factors(phi1)
      FG.add_edges_from([(parts[0], phi1), (parts[1], phi1)])

    bp = BeliefPropagation(FG)
    bp.max_calibrate()

    datadir=GRDATA
    ifile = open(datadir+idx)
    Expln = {}
    for ln in ifile:
      parts = ln.strip().split(",")
      if parts[1]==parts[2]:
          continue
      key = parts[1]+":"+parts[2]
      Expln[key]=float(parts[3])
    ifile.close()
    #print(Expln)

    pre = {}
    for k in Expln.keys():
      parts = k.split(":")
      pvars = []
      for p in parts:
          if p not in FG.get_variable_nodes():
              continue
          pvars.append(p)
      if len(pvars) ==2:
          Q1 = bp.query(variables=pvars)
          pre[k]=Q1.values[1][1]

    for k in Expln.keys():
      parts = k.split(":")
      FG.add_nodes_from(parts)
      phi1 = DiscreteFactor(parts, [2, 2], [1,1,1,math.exp(Expln[k])])
      FG.add_factors(phi1)
      FG.add_edges_from([(parts[0], phi1), (parts[1], phi1)])

    bp = BeliefPropagation(FG)
    bp.max_calibrate()

    post = {}
    for k in Expln.keys():
      parts = k.split(":")
      Q1 = bp.query(variables=parts)
      post[k]=Q1.values[1][1]

    diff = {}
    for k in post.keys():
      parts = k.split(":")
      for k1 in pre:
          parts1 = k1.split(":")
          if len(parts)==len(parts1):
              if len(parts)==2:
                  if (parts[0]==parts1[0] and parts[1]==parts1[1]) or (parts[1]==parts1[0] and parts[0]==parts1[1]):
                      v = abs(post[k]-pre[k])
                      diff[k]=v
                      break
    #print(pre)
    #print(post)
    L = sorted(diff.items(), key=lambda item: item[1],reverse=True)
    #print(L)
    outdir = BPDATA
    ofile = open(outdir+idx,'w')
    for l in L:
      ofile.write(l[0]+","+str(l[1])+"\n")
    ofile.close()


def getIMPList(idx, arg=None, orig=False):
  datadir = f"data/{arg.dataset_name}/{GRDATA}"
  if idx.find(".txt")<0:
      return []
  if not os.path.exists(datadir+idx):
      return []
  ifile = open(datadir+idx)
  edge_imp = {}
  counts = {}
  for ln in ifile:
      parts = ln.strip().split(",")
      key = parts[1]+":"+parts[2]
      if key not in edge_imp:
          edge_imp[key]=float(parts[3])
          counts[key]=1
      else:
          edge_imp[key]=edge_imp[key] + float(parts[3])
          counts[key]=counts[key]+1
  ifile.close()
  for k in edge_imp.keys():
      v = edge_imp[k]/counts[k]
      edge_imp[k] = v
  L = sorted(edge_imp.items(), key=lambda item: item[1],reverse=True)
  return L


#test most important gnn explanation edge
#testpreds = grpreds
#test most important BP explanation edge

def getstats(testpreds,origpreds, arg=None):
  fnames = os.listdir(f"data/{arg.dataset_name}/{BPDATA}")
  nodelist = []
  for f in fnames:
    if f.find(".txt") < 0:
      continue
    idx = int(f[:f.find(".txt")])
    nodelist.append(idx)

  c1=0
  c2=0
  c3=0
  c4=0
  #classvals = [1,2,3]
  classvals = np.zeros(arg.num_classes)
  for lv in classvals:
    for i in nodelist:
    # for i in range(0,len(testpreds),1):
      if origpreds[i]==lv and testpreds[i]==lv:
        c1 = c1+1
      if origpreds[i]==lv and testpreds[i]!=lv:
        c2 = c2+1
      if origpreds[i]!=lv and testpreds[i]==lv:
        c3 = c3+1
      if origpreds[i]!=lv and testpreds[i]!=lv:
        c4 = c4+1
    data = [[c1, c2],[c3, c4]]
    print(str(mcnemar(data, exact=True)))


# # fnames = os.listdir(RANKEXPDATA)
# fnames = os.listdir(BPDATA)
# lcnt = np.zeros(num_classes)
# for f in fnames:
#   if f.find(".txt") < 0:
#     continue
#   idx = int(f[:f.find(".txt")])
#   lcnt[label[idx]] = lcnt[label[idx]] + 1
# print(lcnt, sum(lcnt))

def get_results(G, arg=None):
    model, origpreds = train(GCN, G)

    dirn = f"data/{arg.dataset_name}/{BPDATA}"
    for RM in range(0, 10, 1):
      print(f"RM : {RM}\n")
      A_tmp = pd.read_csv(f"data/{arg.dataset_name}/{adjfile}", header=None, sep='\t').to_numpy()
      for idx in os.listdir(dirn):
        if idx.find(".txt")<0:
          continue
        ifile=open(dirn+idx)
        expimp = {}
        for ln in ifile:
          parts = ln.strip().split(",")
          pval = float(parts[1])
          expimp[parts[0]] = pval
        L = sorted(expimp.items(), key=lambda item: item[1],reverse=True)
        ifile.close()
        if len(L)<=RM:
          continue
        parts = L[RM][0].split(":")
        if A_tmp[int(parts[0])][int(parts[1])]==1:
          A_tmp[int(parts[0])][int(parts[1])]=0
        if A_tmp[int(parts[1])][int(parts[0])]==1:
          A_tmp[int(parts[1])][int(parts[0])]=0
      gr = Graph(arg.dataset_name, G.ndata['feat'], G.ndata['label'], A_tmp)
      # , G.ndata['train_mask'], G.ndata['val_mask'], G.ndata['test_mask'])
      LG = gr.generate()
      model, bppredlist = train(GCN, LG)

      for idx in os.listdir(BPDATA):
        L = getIMPList(idx)
        if RM > len(L)-1:
          break
        parts = L[RM][0].split(":")
        if A_tmp[int(parts[0])][int(parts[1])]==1:
          A_tmp[int(parts[0])][int(parts[1])]=0
        if A_tmp[int(parts[1])][int(parts[0])]==1:
          A_tmp[int(parts[1])][int(parts[0])]=0
      gr = Graph(arg.dataset_name, G.ndata['feat'], G.ndata['label'], A_tmp)
      # , G.ndata['train_mask'], G.ndata['val_mask'], G.ndata['test_mask'])
      LG = gr.generate()
      model, grpredlist = train(GCN, LG)

      # RESULTS PRINT FOR EACH RANK
      print("GNN "+str(RM))
      getstats(grpredlist, origpreds)
      print("BP "+str(RM))
      getstats(bppredlist, origpreds)