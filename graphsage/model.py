import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
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

class MulticlassClassificationLoss(ClassificationLoss):
    def __init__(self, reduction=None):
        super().__init__()
        if reduction is not None:
            self.loss = nn.CrossEntropyLoss(reduction=reduction)
        else:
            self.loss = nn.CrossEntropyLoss()

    def _get_correct(self, outputs):
        return torch.argmax(outputs, dim=1)



class SupervisedGraphSageClassify(nn.Module):

    def __init__(self, num_classes, enc, dim_target):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.loss_func=MulticlassClassificationLoss
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))

        self.fc1 = nn.Linear(2 * enc.embed_dim, enc.embed_dim)
        self.fc2 = nn.Linear(enc.embed_dim, dim_target)

        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        hidden1 = F.relu(self.fc1(embeds.t()))
        return self.fc2(hidden1)
        #
        # scores = self.weight.mm(embeds) # matrix multiplication
        # return scores.t() #transpose

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        # return self.xent(scores, labels.squeeze()) # gold
        return self.loss_func(labels.squeeze(),scores)

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.loss=MulticlassClassificationLoss

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


def load_enzyme_data(name, identity_dim, initializer="None"):
    raw_dir='graph_data'

    indicator_path = raw_dir / f'ENZYMES_graph_indicator.txt'
    edges_path =  raw_dir / f'ENZYMES_A.txt'
    graph_labels_path = raw_dir /f'ENZYMES_graph_labels.txt'
    node_labels_path = raw_dir /f'ENZYMES_node_labels.txt'
    edge_labels_path = raw_dir /f'ENZYMES_edge_labels.txt'
    node_attrs_path = raw_dir /f'ENZYMES_node_attributes.txt'
    edge_attrs_path = raw_dir /f'ENZYMES_edge_attributes.txt'

    unique_node_labels = set()
    unique_edge_labels = set()

    indicator, edge_indicator = [-1], [(-1,-1)]
    graph_nodes = defaultdict(list)
    graph_edges = defaultdict(list)
    node_labels = defaultdict(list)
    edge_labels = defaultdict(list)
    node_attrs = defaultdict(list)
    edge_attrs = defaultdict(list)



    unique_node_labels = set()
    unique_edge_labels = set()

    indicator, edge_indicator = [-1], [(-1,-1)]
    graph_nodes = defaultdict(list)
    # graph_edges = defaultdict(list)
    # node_labels = defaultdict(list)
    # edge_labels = defaultdict(list)
    # node_attrs = defaultdict(list)
    # edge_attrs = defaultdict(list)

    # label_map = {}#
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


    with open(indicator_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            graph_id = int(line)
            indicator.append(graph_id)
            graph_nodes[graph_id].append(i)


    #
    labels=[]
    if node_labels_path.exists():
        with open(node_labels_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                node_label = int(line)
                # unique_node_labels.add(node_label)
                graph_id = indicator[i]
                # node_labels[graph_id].append(node_label)
                # node_map[i]=node_label
                labels[i]=graph_labels[indicator[i]]

    num_nodes=19580
    feature_dim=21
    num_feats = feature_dim
    labels = np.empty((num_nodes,1), dtype=np.int64)

    dim_target=6
    feat_data = np.zeros((num_nodes, num_feats))


    # set initializer method
    if initializer == "1hot":
        feat_data = np.eye(num_nodes)
    elif initializer == "random_normal":
        feat_data = np.random.normal(0, 1, (num_nodes, feature_dim))



    adj_lists = defaultdict(set)

    with open(edges_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            edge = [int(e) for e in line.split(',')]

            adj_lists[edge[0]].add(edge[1])

            # edge_indicator.append(edge)

            # edge[0] is a node id, and it is used to retrieve
            # the corresponding graph id to which it belongs to
            # (see README.txt)
            # graph_id = indicator[edge[0]]

            # graph_edges[graph_id].append(edge)

    return feat_data, labels, adj_lists



def load_cora(feature_dim, initializer="None"):
    num_nodes = 2708
    num_feats = feature_dim
    if initializer == "1hot":
        num_feats = num_nodes
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    if initializer == "None":
        with open("cora/cora.content") as fp:
            for i,line in enumerate(fp):
                info = line.strip().split()
                feat_data[i,:] = map(float, info[1:-1])
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels[i] = label_map[info[-1]]
    else:
        print("Initializing with", initializer)
        with open("cora/cora.content") as fp:
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
    with open("cora/cora.cites") as fp:
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
        for k, v in adj_lists.items():
            feat_data[k, 0] = len(v)
    elif initializer == "pagerank":
        feat_data = np.zeros((num_nodes, 1))
        pagerank = nx.pagerank(G)
        for k, v in pagerank.items():
            feat_data[k, 0] = v
    elif initializer == "eigen_decomposition":
        adj_matrix = nx.to_numpy_array(G)
        w, v = LA.eig(adj_matrix)
        indices = np.argsort(w)
        feat_data = np.zeros((num_nodes, feature_dim))
        for i in range(num_nodes):
            for j in range(feature_dim):
                feat_data[i, j] = v[i, j]

    return feat_data, labels, adj_lists

def run_model(dataset, initializer, seed, epochs, classify="node", batch_size=128, feature_dim=100, identity_dim=50):
    # merge run_cora and run_pubmed
    num_nodes_map = {"cora": 2708, "pubmed": 19717}
    num_classes_map = {"cora": 7, "pubmed": 3}
    enc1_num_samples_map = {"cora": 5, "pubmed": 10}
    enc2_num_samples_map = {"cora": 5, "pubmed": 25}

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
    print("feature dim is", feature_dim)
    features = nn.Embedding(num_nodes, feature_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True, feature_dim=feature_dim, num_nodes=num_nodes, initializer=initializer)
    enc1 = Encoder(features, feature_dim, identity_dim, adj_lists, agg1, gcn=True, cuda=False, initializer=initializer)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), num_nodes, cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = enc1_num_samples_map[dataset]
    enc2.num_samples = enc2_num_samples_map[dataset]

    if(classify=='node'):
        graphsage = SupervisedGraphSage(num_classes, enc2)
    else:
        graphsage= SupervisedGraphSageClassify(num_classes,enc2, 6)
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




def run_cora(initializer, seed, epochs, batch_size=128, feature_dim=100, identity_dim=50):
    np.random.seed(seed)
    random.seed(seed)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora(num_nodes, feature_dim, initializer)
    print(feat_data)
    if initializer == "1hot":
        feature_dim = num_nodes
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

def load_pubmed(feature_dim, initializer):
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = feature_dim
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
            feat_data = extract_deepwalk_embeddings("pubmed/pubmed.embeddings", node_map)

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
        adj_matrix = nx.to_numpy_array(G)
        print("start computing eigen vectors")
        w, v = LA.eig(adj_matrix)
        print("finished computing eigen vectors")
        indices = np.argsort(w)
        feat_data = np.zeros((num_nodes, feature_dim))
        for i in range(num_nodes):
            for j in range(feature_dim):
                feat_data[i, j] = v[i, j]

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

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
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


    run_model(dataset, initializer, seed, epochs, classify=classify, feature_dim=feature_dim, identity_dim=identity_dim)

if __name__ == "__main__":
    main()
