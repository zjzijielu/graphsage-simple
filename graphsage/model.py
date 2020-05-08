import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

import argparse
import networkx as nx
from numpy import linalg as LA

import torch.nn.functional as F

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSageClassify(nn.Module):

    # def __init__(self, num_classes, enc, dim_target):
    def __init__(self, num_classes, enc):

        super(SupervisedGraphSageClassify, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        #num_classes: graph calsses
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))

        # self.fc1 = nn.Linear(2 * enc.embed_dim, enc.embed_dim)
        self.fc2 = nn.Linear(enc.embed_dim, 6)#harcode for enzyme, 6 classes

        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        #aggregate the embeddings
        # embeds=embeds.sum(1)
        # exit()
        embeds=torch.mean(embeds,1)
        embeds=embeds.view(128,1)
        # print(embeds)
        
        scores = torch.sigmoid(self.weight.mm(embeds))
        # scores=torch.sigmoid(self.fc2(embeds.t()))
        return scores.t()


    #feed a single graph label at a time
    def loss(self, nodes, labels):
        print(nodes)
        # exit()
        scores = self.forward(nodes)
        # t=labels
        # s=self.xent(scores, labels.squeeze())
        return self.xent(scores, labels)

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def extract_deepwalk_embeddings(filename, node_map):
    with open(filename) as f:
        feat_data = []
        for i, line in enumerate(f):
            info = line.strip().split()
            if i == 0:
                feat_data = np.zeros((int(info[0]), int(info[1])))
            else:
                idx = node_map[info[0]]
                feat_data[idx, :] = list(map(float, info[1::]))
            
    return feat_data


def run_model(dataset, initializer, seed, epochs, classify="node", batch_size=128, feature_dim=100, identity_dim=50):
    # merge run_cora and run_pubmed
    num_nodes_map = {"cora": 2708, "pubmed": 19717}
    num_classes_map = {"cora": 7, "pubmed": 3}
    enc1_num_samples_map = {"cora": 5, "pubmed": 10}
    enc2_num_samples_map = {"cora": 5, "pubmed": 25}
    attribute_dim = {"cora": 1433, "pubmed": 500}

    np.random.seed(seed)
    random.seed(seed)
    feat_data = []
    labels = []
    adj_lists = []
    num_nodes = num_nodes_map[dataset]
    num_classes = num_classes_map[dataset]

    if dataset == "cora":
        feat_data, labels, adj_lists = load_cora(feature_dim, initializer)
    elif dataset == "pubmed":
        feat_data, labels, adj_lists = load_pubmed(feature_dim, initializer)

    # feat_data, labels, adj_lists = load_data(dataset, feature_dim, initializer)
    print(feat_data)
    if initializer == "1hot":
        feature_dim = num_nodes
    elif initializer == "node_degree":
        feature_dim = feat_data.shape[1]
    print("feature dim is", feature_dim)
    features = nn.Embedding(num_nodes, feature_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True, feature_dim=feature_dim, num_nodes=num_nodes, initializer=initializer)
    enc1 = Encoder(features, feature_dim if initializer != "None" else attribute_dim[dataset], identity_dim, adj_lists, agg1, gcn=True, cuda=False, initializer=initializer)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), num_nodes, cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = enc1_num_samples_map[dataset]
    enc2.num_samples = enc2_num_samples_map[dataset]


    graphsage = SupervisedGraphSage(num_classes, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test_end_idx = int(0.1 * num_nodes)
    val_end_idx = int(0.2 * num_nodes)
    test = rand_indices[:test_end_idx]
    val = rand_indices[test_end_idx:val_end_idx]
    train = list(rand_indices[val_end_idx:])
    train_num = len(train)

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []

    for epoch in range(epochs):
        print("Epoch:", epoch)
        random.shuffle(train)
        for batch in range(0, train_num, batch_size):
            batch_nodes = train[batch:max(train_num, batch+batch_size)]
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
            if (batch == 0):
                print("Batch", batch, "Loss:", loss.item())

    val_output = graphsage.forward(val) 
    print("Validation F1 micro:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Validation F1 macro:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="macro"))
    print("Average batch time:", np.mean(times))

def load_cora(feature_dim, initializer="None"):
    num_nodes = 2708
    num_feats = feature_dim if initializer != 'None' else 1433
    if initializer == "1hot":
        num_feats = num_nodes
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    if initializer == "None":
        with open("../cora/cora.content") as fp:
            for i,line in enumerate(fp):
                info = line.strip().split()
                feat_data[i,:] = list(map(float, info[1:-1]))
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels[i] = label_map[info[-1]]
    else:
        print("Initializing with", initializer)
        with open("../cora/cora.content") as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels[i] = label_map[info[-1]]
        # set initializer method
        if initializer == "1hot":
            feat_data = np.eye(num_nodes)
        elif initializer == "random_normal":
            feat_data = np.random.normal(0, 1, (num_nodes, feature_dim))
        elif initializer == "shared":
            feat_data = np.ones((num_nodes, feature_dim))
        elif initializer == "node_degree":
            feat_data = np.zeros((num_nodes, 1))
        elif initializer == "pagerank" or initializer == "eigen_decomposition":
            G = nx.Graph()
            G.add_nodes_from(node_map.values())
        elif initializer == "deepwalk":
            feat_data = extract_deepwalk_embeddings("cora/cora.embeddings", node_map)

    adj_lists = defaultdict(set)
    with open("../cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
            if initializer == "pagerank" or initializer == "eigen_decomposition":
                G.add_edge(paper1, paper2)
                G.add_edge(paper2, paper1)

    if initializer == "node_degree":
        # convert to 1hot representation
        node_degrees = [len(v) for v in adj_lists.values()]
        max_degree = max(node_degrees)
        feat_data = np.zeros((num_nodes, max_degree+1))
        for k, v in adj_lists.items():
            feat_data[k, len(v)] = 1
    elif initializer == "pagerank":
        feat_data = np.zeros((num_nodes, 1))
        pagerank = nx.pagerank(G)
        for k, v in pagerank.items():
            feat_data[k, 0] = v
    elif initializer == "eigen_decomposition":
        try:
            v = np.load("cora/cora_eigenvector.npy")
            print(v.shape)
        except:
            adj_matrix = nx.to_numpy_array(G)
            print("start computing eigen vectors")
            w, v = LA.eig(adj_matrix)
            indices = np.argsort(w)[::-1]
            v = v.transpose()[indices]
            # only save top 1000 eigenvectors
            np.save("cora/cora_eigenvector", v[:1000])
        print(v)
        feat_data = np.zeros((num_nodes, feature_dim))
        assert(feature_dim <= 1000)
        for i in range(num_nodes):
            for j in range(feature_dim):
                feat_data[i, j] = v[j, i]

    return feat_data, labels, adj_lists


def run_cora(initializer, seed, epochs, batch_size=128, feature_dim=100, identity_dim=50):
    np.random.seed(seed)
    random.seed(seed)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora( 2708, "1hot")
    print(feat_data, initializer)
    if initializer == "1hot":
        feature_dim = num_nodes
    elif initializer == "node_degree":
        feature_dim = feat_data.shape[1]
        print(feat_data.shape)
    print("feature dim is", feature_dim)
    features = nn.Embedding(num_nodes, feature_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True, feature_dim=feature_dim)
    enc1 = Encoder(features, feature_dim, identity_dim, adj_lists, agg1, gcn=True, cuda=False, initializer=initializer)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)

    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:271]
    val = rand_indices[271:542]
    train = list(rand_indices[542:])
    train_num = len(train)

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []

    for epoch in range(epochs):
        print("Epoch:", epoch)
        random.shuffle(train)
        for batch in range(0, train_num, batch_size):
            batch_nodes = train[batch:max(train_num, batch+batch_size)]
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
            if (batch == 0):
                print("Batch", batch, "Loss:", loss.item())

    val_output = graphsage.forward(val) 
    print("Validation F1 micro:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Validation F1 macro:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="macro"))
    print("Average batch time:", np.mean(times))



def parse_tu_data(name, raw_dir):
    # setup paths
    # indicator_path = raw_dir / name / f'{name}_graph_indicator.txt'
    # edges_path = raw_dir / name / f'{name}_A.txt'
    # graph_labels_path = raw_dir / name / f'{name}_graph_labels.txt'
    # node_labels_path = raw_dir / name / f'{name}_node_labels.txt'
    # edge_labels_path = raw_dir / name / f'{name}_edge_labels.txt'
    # node_attrs_path = raw_dir / name / f'{name}_node_attributes.txt'
    # edge_attrs_path = raw_dir / name / f'{name}_edge_attributes.txt'

    # indicator_path = 'graph_data/ENZYMES_graph_indicator.txt'
    # edges_path = 'graph_data/ENZYMES_A.txt'
    # graph_labels_path ='graph_data/ENZYMES_graph_labels.txt'
    # node_labels_path = 'graph_data/ENZYMES_node_labels.txt'
    # edge_labels_path = 'graph_data/ENZYMES_edge_labels.txt'
    # node_attrs_path = 'graph_data/ENZYMES_node_attributes.txt'
    # edge_attrs_path ='graph_data/ENZYMES_edge_attributes.txt'
    indicator_path = '../graph_data/ENZYMES_graph_indicator.txt'
    edges_path = '../graph_data/ENZYMES_A.txt'
    graph_labels_path ='../graph_data/ENZYMES_graph_labels.txt'
    node_labels_path = '../graph_data/ENZYMES_node_labels.txt'
    edge_labels_path = '../graph_data/ENZYMES_edge_labels.txt'
    node_attrs_path = '../graph_data/ENZYMES_node_attributes.txt'
    edge_attrs_path ='../graph_data/ENZYMES_edge_attributes.txt'
    unique_node_labels = set()
    unique_edge_labels = set()

    indicator, edge_indicator = [-1], [(-1,-1)]
    graph_nodes = defaultdict(list)
    graph_edges = defaultdict(list)
    node_labels = defaultdict(list)
    edge_labels = defaultdict(list)
    node_attrs = defaultdict(list)
    edge_attrs = defaultdict(list)

    with open(indicator_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            graph_id = int(line)
            indicator.append(graph_id)
            graph_nodes[graph_id].append(i)
    adj_lists = defaultdict(set)
    with open(edges_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            edge = [int(e) for e in line.split(',')]
            if edge[0] in adj_lists:
                adj_lists[edge[0]].add(edge[1])
            else:
                adj_lists[edge[0]]=set()
            edge_indicator.append(edge)


            # edge[0] is a node id, and it is used to retrieve
            # the corresponding graph id to which it belongs to
            # (see README.txt)
            graph_id = indicator[edge[0]]

            graph_edges[graph_id].append(edge)
        # t=adj_lists[2]
        # u=adj_lists[3]
        # s=1
    # if node_labels_path.exists():
    with open(node_labels_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            node_label = int(line)
            unique_node_labels.add(node_label)
            graph_id = indicator[i]
            node_labels[graph_id].append(node_label)

    # if edge_labels_path.exists():
    # with open(edge_labels_path, "r") as f:
    #     for i, line in enumerate(f.readlines(), 1):
    #         line = line.rstrip("\n")
    #         edge_label = int(line)
    #         unique_edge_labels.add(edge_label)
    #         graph_id = indicator[edge_indicator[i][0]]
    #         edge_labels[graph_id].append(edge_label)

    # if node_attrs_path.exists():
    with open(node_attrs_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            nums = line.split(",")
            node_attr = np.array([float(n) for n in nums])
            graph_id = indicator[i]
            node_attrs[graph_id].append(node_attr)

    # if edge_attrs_path.exists():
    # with open(edge_attrs_path, "r") as f:
    #     for i, line in enumerate(f.readlines(), 1):
    #         line = line.rstrip("\n")
    #         nums = line.split(",")
    #         edge_attr = np.array([float(n) for n in nums])
    #         graph_id = indicator[edge_indicator[i][0]]
    #         edge_attrs[graph_id].append(edge_attr)

    # get graph labels
    graph_labels = []
    with open(graph_labels_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            target = int(line)
            if target == -1:
                graph_labels.append(0)
            else:
                graph_labels.append(target)

        if min(graph_labels) == 1:  # Shift by one to the left. Apparently this is necessary for multiclass tasks.
            graph_labels = [l - 1 for l in graph_labels]

    num_node_labels = max(unique_node_labels) if unique_node_labels != set() else 0
    if num_node_labels != 0 and min(unique_node_labels) == 0:  # some datasets e.g. PROTEINS have labels with value 0
        num_node_labels += 1

    num_edge_labels = max(unique_edge_labels) if unique_edge_labels != set() else 0
    if num_edge_labels != 0 and min(unique_edge_labels) == 0:
        num_edge_labels += 1

    return {
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "graph_labels": graph_labels,
        "node_labels": node_labels,
        "node_attrs": node_attrs,
        "edge_labels": edge_labels,
        "edge_attrs": edge_attrs
    }, num_node_labels, num_edge_labels, adj_lists



def load_enzyme(feature_dim, initializer):
    num_nodes=19474
    num_feats = feature_dim if initializer != 'None' else 500
    if initializer == "1hot":
        num_feats = num_nodes
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    # node_map = {}
    # label_map = {}
    graphs_data, num_node_labels, num_edge_labels,adj_lists= parse_tu_data("ENZYMES", "graph_data")
    labels=graphs_data["node_labels"]
    if initializer == "1hot":
        feat_data = np.eye(num_nodes)
    elif initializer == "random_normal":
        feat_data = np.random.normal(0, 1, (num_nodes, feature_dim))
    elif initializer == "shared":
        feat_data = np.ones((num_nodes, feature_dim))
    elif initializer == "node_degree":
        feat_data = np.zeros((num_nodes, 1))
        for k, v in adj_lists.items():
            feat_data[k, 0] = len(v)
    return graphs_data, num_edge_labels, num_edge_labels, feat_data, labels, adj_lists

#

def run_enzyme(feature_dim,initializer,identity_dim=50):
    graphs_data, num_edge_labels, num_edge_labels, feat_data, labels, adj_lists=load_enzyme(feature_dim,initializer)
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19474
    # graphs_data, num_edge_labels, num_edge_labels, feat_data, labels, adj_lists=load_enzyme()
    features = nn.Embedding(19474, 19474)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    graph_nodes=graphs_data["graph_nodes"]



    agg1 = MeanAggregator(features, cuda=True, feature_dim=feature_dim, num_nodes=num_nodes, initializer=initializer)
    enc1 = Encoder(features, feature_dim, identity_dim, adj_lists,
                   agg1, gcn=True, cuda=False, initializer=initializer)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), num_nodes, cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25
    graphsage = SupervisedGraphSageClassify(6, enc2)#hardcode
    # filtered = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,
    #             37,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,
    #             68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]
    total=np.arange(600)
    random.shuffle(total)
    ##################### whole, some graph returns nan embedding #####################
    train=total[1:500]
    val=total[500:550]
    test=total[550:600]
    ##################### filtered #####################
    # train=filtered[0:9]
    # val=filtered[10:15]
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.1)
    # do not do batch, feed graph one at a time
    for epoch in range(1):#harcode: 10 epochs
        for i in train:
        # for i in range(59,100):
            graph_nodes=graphs_data["graph_nodes"][i] #todo debug graph nodes, id map

            optimizer.zero_grad()
            graph_label=np.array([graphs_data['graph_labels'][i]-1])
            # loss = graphsage.loss(graph_nodes,
            #                       Variable(torch.LongTensor(graph_label)))#todo, debug lables,
            # res=graphsage.forward(graph_nodes)

            loss = graphsage.loss(graph_nodes,
                                  Variable(torch.LongTensor(graph_label)))#todo, debug lables,

            loss.backward()
            optimizer.step()
            end_time = time.time()
            print(i, loss)


    # val_output = graphsage.forward(val)
    # res=val_output.data.numpy().argmax(axis=1)
    # true_label=graphs_data['graph_labels'][val]-1
    # # print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    # print("Validation F1:", f1_score(graphs_data['graph_labels'][val]-1, val_output.data.numpy().argmax(axis=1), average="micro"))
    #
    # print("Average batch time:", np.mean(times))
    true_label=[]
    for t in val:
        true_label.append(graphs_data['graph_labels'][t]-1)
    # lbs=graphs_data['graph_labels'][val]
    # true_label = graphs_data['graph_labels'][val] - [1]
    all_val_res=[]#the predicted
    for v in val:
        val_output = graphsage.forward(graphs_data["graph_nodes"][v])
        res=val_output.data.numpy().argmax(axis=1)
        all_val_res.append(res[0])


    # print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Validation F1:", f1_score(true_label, all_val_res, average="micro"))

    # print("Average batch time:", np.mean(times))







def load_pubmed(feature_dim, initializer):
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = feature_dim if initializer != 'None' else 500
    if initializer == "1hot":
        num_feats = num_nodes
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    if initializer == "None":
        with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
            fp.readline()
            feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
            for i, line in enumerate(fp):
                info = line.split("\t")
                node_map[info[0]] = i
                labels[i] = int(info[1].split("=")[1])-1
                for word_info in info[2:-1]:
                    word_info = word_info.split("=")
                    feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    else:
        with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
            fp.readline()
            fp.readline()
            for i, line in enumerate(fp):
                info = line.split("\t")
                node_map[info[0]] = i
                labels[i] = int(info[1].split("=")[1])-1

        # set initializer method
        if initializer == "1hot":
            feat_data = np.eye(num_nodes)
        elif initializer == "random_normal":
            feat_data = np.random.normal(0, 1, (num_nodes, feature_dim))
        elif initializer == "shared":
            feat_data = np.ones((num_nodes, feature_dim))
        elif initializer == "node_degree":
            feat_data = np.zeros((num_nodes, 1))
        elif initializer == "pagerank" or initializer == "eigen_decomposition":
            G = nx.Graph()
            G.add_nodes_from(node_map.values())
        elif initializer == "deepwalk":
            feat_data = extract_deepwalk_embeddings("pubmed-data/Pubmed.embeddings", node_map)

    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
            if initializer == "pagerank" or initializer == "eigen_decomposition":
                G.add_edge(paper1, paper2)
                G.add_edge(paper2, paper1)

    if initializer == "node_degree":
        for k, v in adj_lists.items():
            feat_data[k, 0] = len(v)
    elif initializer == "pagerank":
        feat_data = np.zeros((num_nodes, 1))
        pagerank = nx.pagerank(G)
        for k, v in pagerank.items():
            feat_data[k, 0] = v
    elif initializer == "eigen_decomposition":
        try:
            v = np.load("pubmed-data/pubmed_eigenvector.npy")
            # print(v.shape)
        except:
            adj_matrix = nx.to_numpy_array(G)
            print("start computing eigen vectors")
            w, v = LA.eig(adj_matrix)
            indices = np.argsort(w)[::-1]
            v = v.transpose()[indices]
            # only save top 1000 eigenvectors
            np.save("pubmed-data/pubmed_eigenvector", v[:1000])
        print(v)
        feat_data = np.zeros((num_nodes, feature_dim))
        assert(feature_dim <= 1000)
        for i in range(num_nodes):
            for j in range(feature_dim):
                feat_data[i, j] = v[j, i]

    return feat_data, labels, adj_lists

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False, initializer=initializer)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.0001)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        t=type(batch_nodes)
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.data[0])

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initializer", type=str, default="None",
                        help="node feature initialiation method")
    parser.add_argument("--identity_dim", type=int, default=50,
                        help="node embedding dimension")
    parser.add_argument("--feature_dim", type=int, default=100,
                        help="node feature dimension")
    parser.add_argument("--seed", type=int, default="1",
                        help="random seed for initialization")
    parser.add_argument("--epochs", type=int, default="5",
                        help="random seed for initialization")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="dataset used")
    parser.add_argument("--classify", type=str, default="node",
                        help="classify task")

    args = parser.parse_args()

    initializer = args.initializer
    identity_dim = args.identity_dim
    feature_dim = args.feature_dim
    seed = args.seed
    epochs = args.epochs
    dataset = args.dataset
    classify = args.classify

    run_enzyme(19474,"1hot")
    #


    # run_model(dataset, initializer, seed, epochs, classify=classify, feature_dim=feature_dim, identity_dim=identity_dim)

if __name__ == "__main__":
     main()
    # run_cora("1hot",1,10,50,100,50)
# --initializer "1hot" --identity_dim 50 --feature_dim 100 --seed 1 --epochs 5 --dataset "cora" --classify "node"