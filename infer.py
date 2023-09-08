import pandas as pd
import numpy as np
import scipy
from collections import Counter

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import random
import math
import pickle as pk

import seaborn
import scipy.sparse as sp
from matplotlib import pyplot

import dgl
dgl.backend.set_default_backend(default_dir='pytorch', backend_name='pytorch')

import dgl.function as fn
from dgl import DGLGraph
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn import GNNExplainer

import networkx as nx

import Graph
import GCN

gnn = 'GCN'
gnn_explainer = "GNN-Explainer"

featurefile = "feat.txt"
adjfile = "A.txt"
labelfile = "label.txt"
trainmaskfile = "train_mask.txt"
validmaskfile = "val_mask.txt"
testmaskfile = "test_mask.txt"

train_mask = None
valid_mask = None
test_mask = None

GRDATA = "grexpTexas/"
BMFDATA = "BMFTexas/"
BPDATA = "BPProbsTexas/"
RANKEXPDATA = "rankexpTexas/"

dataset_name = 'texas'

#PARAMETERS
STARTRANK = 140
ENDRANK = 181
INCRANK = 10
EXPLIMITNODES = 1
CLASSVALS = [0,1,2,3,4]




dirn = GRDATA
model,preds = train(GCN, G)
explainer = GNNExplainer(model, num_hops=2, log=False)
for idx in range(G.num_nodes()):
    # if label[idx] ==0:
    nc, sg, fm, em = explainer.explain_node(node_id = idx, graph=G, feat = G.ndata['feat'])
    if len(sg.nodes()) > EXPLIMITNODES:
        nodes = sg.ndata[dgl.NID].tolist()
        ofile = open(dirn+str(idx)+".txt",'w')
        src, dst = sg.edges()
        for s, d, em in zip(src.numpy(), dst.numpy(), em.numpy()):
            s1 = str(nodes[s])
            d1 = str(nodes[d])
            ofile.write("-1"+","+s1+","+d1+","+str(em)+"\n")
        ofile.close()

PREFIX = BMFDATA
dirn = RANKEXPDATA

for r in range(STARTRANK,ENDRANK,INCRANK):
    _D = pd.read_csv(PREFIX+str(r)+'.txt', header=None, sep=',').to_numpy()
    gr = Graph(dataset_name, G.ndata['feat'], G.ndata['label'], _D, G.ndata['train_mask'], G.ndata['val_mask'], G.ndata['test_mask'])
    LG = gr.generate()
    model,lrpreds = train(GCN, LG)
    explainer = GNNExplainer(model, num_hops=2, log=False)
    for idx in range(G.num_nodes()):
        # if label[idx] ==0:
        nc, sg, fm, em = explainer.explain_node(node_id = idx, graph=LG, feat = LG.ndata['feat'])
        if len(sg.nodes()) > EXPLIMITNODES:
            nodes = sg.ndata[dgl.NID].tolist()
            ofile = open(dirn+str(idx)+".txt",'a')
            src, dst = sg.edges()
            for s, d, em in zip(src.numpy(), dst.numpy(), em.numpy()):
                s1 = str(nodes[s])
                d1 = str(nodes[d])
                ofile.write(str(r)+","+s1+","+d1+","+str(em)+"\n")
            ofile.close()
    print("done "+str(r))

import os
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import math
import numpy as np

idnames = os.listdir(RANKEXPDATA)
LR = []
GR = []
#ofile = open("bpresults.txt",'w')
def getBPList(idx):
    datadir = RANKEXPDATA
    if idx.find(".txt")<0:
        return []
    if not os.path.exists(datadir+idx):
        return []
    if not os.path.exists(GRDATA+idx):
        return
    ifile = open(datadir+idx)
    edge_imp = {}
    counts = {}
    allnodes = {}
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
        #print(str(k)+" "+str(v))
    #sorted(edge_imp.items(), key=lambda item: item[1],reverse=True)
    #print(idx)
    L = sorted(edge_imp.items(), key=lambda item: item[1],reverse=True)
    #print(L)
    #return L
    FG = FactorGraph()

    if len(edge_imp) < 100:
      for k in edge_imp.keys():
          parts = k.split(":")
          FG.add_nodes_from(parts)
          phi1 = DiscreteFactor(parts, [2, 2], [1,1,1,math.exp(edge_imp[k])])
          #phi1 = DiscreteFactor(parts, [2, 2], [0.1,0.1,0.1,edge_imp[k]])
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
      #sdict = {}
      #print(Expln)
      pre = {}
      for k in Expln.keys():
          parts = k.split(":")
          Vars = []
          for p in parts:
              if p not in FG.get_variable_nodes():
                  continue
              Vars.append(p)
          if len(Vars) ==2:
              Q1 = bp.query(variables=Vars)
              pre[k]=Q1.values[1][1]

      for k in Expln.keys():
          parts = k.split(":")
          FG.add_nodes_from(parts)
          phi1 = DiscreteFactor(parts, [2, 2], [1,1,1,math.exp(Expln[k])])
          #phi1 = DiscreteFactor(parts, [2, 2], [0.1,0.1,0.1,edge_imp[k]])
          FG.add_factors(phi1)
          FG.add_edges_from([(parts[0], phi1), (parts[1], phi1)])

      bp = BeliefPropagation(FG)
      bp.max_calibrate()
      post = {}
      for k in Expln.keys():
          parts = k.split(":")
          Q1 = bp.query(variables=parts)
          #print(str(parts)+" "+str(Q1.values))
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

      '''
      for k in Expln.keys():
          Evid = k.split(":")
          Vars = []
          for e in Evid:
              if e not in FG.get_variable_nodes():
                  continue
              Vars.append(e)
          if len(Vars)>0:
              Q1 = bp.query(variables=Vars)
              if len(Vars)==1:
                  ofile.write(Vars[0]+":"+str(Q1.values[0])+","+str(Q1.values[1])+"\n")
              elif len(Vars)==2:
                  ofile.write(Vars[0]+","+Vars[1]+":"+str(Q1.values[0][0])+
                              ","+str(Q1.values[0][1])+","+str(Q1.values[1][0])+","+str(Q1.values[1][1])+"\n")

      '''
      '''
      for k in Expln.keys():
          Vars = [idx[:idx.find(".txt")]]
          Evid = k.split(":")
          Edict = {}
          inVar = {}
          for e in Evid:
              if e not in FG.get_variable_nodes():
                  continue
              if e in Vars:
                  inVar[e] = True
                  continue
              Edict[e]=0
              #ofile.write(str(e)+":0,")
          Q1 = bp.query(variables=Vars,evidence=Edict)
          #ofile.write(str(Q1.values[0])+":"+str(Q1.values[1])+"\n")
          for e in Evid:
              if e not in FG.get_variable_nodes():
                  continue
              if e in Vars:
                  continue
              Edict[e]=1
              #ofile.write(str(e)+":1,")
          Q2 = bp.query(variables=Vars,evidence=Edict)
          #ofile.write(str(Q2.values[0])+":"+str(Q2.values[1])+"\n")
          #diff = abs(Q1.values[1]-Q2.values[1])
          #sdict[k]=diff
          allvars = []
          for e in inVar.keys():
              allvars.append(e)
          for e in Edict:
              allvars.append(e)
          print(allvars)
          #for v in Vars:
          #    print(v+" ")
          #print(Vars)
          #print(Edict)
          print(str(Q1.values)+ " "+str(Q2.values))
      '''
      '''
      Vars = []
      for k in Expln.keys():
          var = k.split(":")
          for v in var:
                  if v not in Vars:
                      Vars.append(v)

      if len(Vars)>0:
          for v in Vars:
              if v not in FG.get_variable_nodes():
                  continue
              Edict = {}
              Edict[v]=0
              Vars1 = []
              for v1 in Vars:
                  if v==v1:
                      continue
                  if v1 not in FG.get_variable_nodes():
                      continue

                  Vars1.append(v1)
              Q = bp.query(variables=Vars1,evidence=Edict)
              print(Vars1)
              print(Edict)
              print(Q.values)

      '''
      #L = sorted(sdict.items(), key=lambda item: item[1],reverse=True)
      #return L
      ofile.close()


for idx in idnames:
    getBPList(idx)

model,origpreds = train(GCN, G)

import os
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar


def getIMPList(idx,orig=False):
  datadir = GRDATA
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

def getstats(testpreds,origpreds):
  fnames = os.listdir(BPDATA)
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
  classvals = CLASSVALS
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


# fnames = os.listdir(RANKEXPDATA)
fnames = os.listdir(BPDATA)
lcnt = np.zeros(num_classes)
for f in fnames:
  if f.find(".txt") < 0:
    continue
  idx = int(f[:f.find(".txt")])
  lcnt[label[idx]] = lcnt[label[idx]] + 1
print(lcnt, sum(lcnt))

for RM in range(0,10,1):
  print(f"RM : {RM}\n")
  A_tmp = pd.read_csv(adjfile, header=None, sep='\t').to_numpy()
  for idx in os.listdir(BPDATA):
    if idx.find(".txt")<0:
      continue
    ifile=open(BPDATA+idx)
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
  gr = Graph(dataset_name, G.ndata['feat'], G.ndata['label'], A_tmp)
  # , G.ndata['train_mask'], G.ndata['val_mask'], G.ndata['test_mask'])
  LG = gr.generate()
  model,bppredlist = train(GCN, LG)

  for idx in os.listdir(BPDATA):
    L = getIMPList(idx)
    if RM > len(L)-1:
      break
    parts = L[RM][0].split(":")
    if A_tmp[int(parts[0])][int(parts[1])]==1:
      A_tmp[int(parts[0])][int(parts[1])]=0
    if A_tmp[int(parts[1])][int(parts[0])]==1:
      A_tmp[int(parts[1])][int(parts[0])]=0
  gr = Graph(dataset_name, G.ndata['feat'], G.ndata['label'], A_tmp)
  # , G.ndata['train_mask'], G.ndata['val_mask'], G.ndata['test_mask'])
  LG = gr.generate()
  model,grpredlist = train(GCN, LG)

  # RESULTS PRINT FOR EACH RANK
  print("GNN "+str(RM))
  getstats(grpredlist,origpreds)
  print("BP "+str(RM))
  getstats(bppredlist,origpreds)