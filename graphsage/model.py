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

import matplotlib.pyplot as plt

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

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

def extract_deepwalk_embeddings(filename, node_map, dataset="cora"):
    with open(filename) as f:
        feat_data = []
        for i, line in enumerate(f):
            info = line.strip().split()
            if i == 0:
                feat_data = np.zeros((int(info[0]), int(info[1])))
            else:
                idx = None
                if dataset != "citeseer":
                    idx = node_map[info[0]]
                else:
                    idx = int(info[0])
                feat_data[idx, :] = list(map(float, info[1::]))
            
    return feat_data

def load_brazil_airport(feature_dim, initializer="None"):
    '''
    hardcoded for simplicity
    '''
    num_nodes = 131
    num_feats = feature_dim if initializer != 'None' else 1433
    if initializer == "1hot":
        num_feats = num_nodes
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    if initializer == "None":
        with open("brazil-airports/labels-brazil-airports.txt") as fp:
            for i,line in enumerate(fp):
                info = line.strip().split()
                feat_data[i,:] = list(map(float, info[1:-1]))
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                    print(info[-1])
                labels[i] = label_map[info[-1]]
    else:
        print("Initializing with", initializer)
        with open("brazil-airports/labels-brazil-airports.txt") as fp:
            fp.readline()
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
            feat_data = extract_deepwalk_embeddings("brazil-airports/brazil-airport.embeddings", node_map)

    adj_lists = defaultdict(set)
    with open("brazil-airports/brazil-airports.edgelist") as fp:
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
            v = np.load("brazil-airports/brazil-airport_eigenvector.npy")
            print(v.shape)
        except:
            adj_matrix = nx.to_numpy_array(G)
            print(adj_matrix.shape)
            print("start computing eigen vectors")
            w, v = LA.eig(adj_matrix)
            indices = np.argsort(w)[::-1]
            v = v.transpose()[indices]
            # only save top 1000 eigenvectors
            np.save("brazil-airports/brazil-airport_eigenvector", v[:max(1000, num_nodes)])

        feat_data = np.zeros((num_nodes, feature_dim))
        assert(feature_dim <= 1000)
        for i in range(num_nodes):
            for j in range(feature_dim):
                feat_data[i, j] = v[j, i]

    return feat_data, labels, adj_lists

def load_usa_airport(feature_dim, initializer="None"):
    '''
    hardcoded for simplicity
    '''
    num_nodes = 1190
    num_feats = feature_dim if initializer != 'None' else 1433
    if initializer == "1hot":
        num_feats = num_nodes
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    if initializer == "None":
        with open("usa-airport/labels-usa-airports.txt") as fp:
            for i,line in enumerate(fp):
                info = line.strip().split()
                feat_data[i,:] = list(map(float, info[1:-1]))
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                    print(info[-1])
                labels[i] = label_map[info[-1]]
    else:
        print("Initializing with", initializer)
        with open("usa-airport/labels-usa-airports.txt") as fp:
            fp.readline()
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
            feat_data = extract_deepwalk_embeddings("usa-airport/usa-airport.embeddings", node_map)

    adj_lists = defaultdict(set)
    with open("usa-airport/usa-airports.edgelist") as fp:
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
            v = np.load("usa-airport/usa-airport_eigenvector.npy")
            print(v.shape)
        except:
            adj_matrix = nx.to_numpy_array(G)
            print(adj_matrix.shape)
            print("start computing eigen vectors")
            w, v = LA.eig(adj_matrix)
            indices = np.argsort(w)[::-1]
            v = v.transpose()[indices]
            # only save top 1000 eigenvectors
            np.save("usa-airport/usa-airport_eigenvector", v[:1000])

        feat_data = np.zeros((num_nodes, feature_dim))
        assert(feature_dim <= 1000)
        for i in range(num_nodes):
            for j in range(feature_dim):
                feat_data[i, j] = v[j, i]

    return feat_data, labels, adj_lists

def load_europe_airport(feature_dim, initializer="None"):
    '''
    hardcoded for simplicity
    '''
    num_nodes = 399
    num_feats = feature_dim
    if initializer == "1hot":
        num_feats = num_nodes
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    if initializer == "None":
        with open("europe-airports/labels-europe-airports.txt") as fp:
            for i,line in enumerate(fp):
                info = line.strip().split()
                feat_data[i,:] = list(map(float, info[1:-1]))
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                    print(info[-1])
                labels[i] = label_map[info[-1]]
    else:
        print("Initializing with", initializer)
        with open("europe-airports/labels-europe-airports.txt") as fp:
            fp.readline()
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

            feat_data = extract_deepwalk_embeddings("europe-airports/europe-airports.embeddings", node_map)
    adj_lists = defaultdict(set)
    with open("europe-airports/europe-airports.edgelist") as fp:
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
            v = np.load("europe-airports/europe-airport_eigenvector.npy")
            print(v.shape)
        except:
            adj_matrix = nx.to_numpy_array(G)
            print(adj_matrix.shape)
            print("start computing eigen vectors")
            w, v = LA.eig(adj_matrix)
            indices = np.argsort(w)[::-1]
            v = v.transpose()[indices]
            # only save top 1000 eigenvectors
            np.save("europe-airports/europe-airport_eigenvector", v[:max(1000, num_nodes)])

        feat_data = np.zeros((num_nodes, feature_dim))
        assert(feature_dim <= 1000)
        for i in range(num_nodes):
            for j in range(feature_dim):
                feat_data[i, j] = v[j, i]

    return feat_data, labels, adj_lists

def load_citeseer(feature_dim, initializer="None"):
    '''
    hard coded for simplicity
    '''

    num_nodes = 3312
    num_feats = feature_dim if initializer != 'None' else 3703
    if initializer == "1hot":
        num_feats = num_nodes
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    if initializer == "None":
        with open("citeseer/citeseer.content") as fp:
            for i,line in enumerate(fp):
                info = line.strip().split()
                # print(len(list(map(float, info[1:-1]))))
                feat_data[i,:] = list(map(float, info[1:-1]))
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels[i] = label_map[info[-1]]
    else:
        print("Initializing with", initializer)
        with open("citeseer/citeseer.content") as fp:
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
            feat_data = extract_deepwalk_embeddings("citeseer/citeseer.embeddings", node_map, "citeseer")

    adj_lists = defaultdict(set)
    with open("citeseer/citeseer.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            try:
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
            except:
                # print(info[0], info[1])
                continue
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
            v = np.load("citeseer/citeseer_eigenvector.npy")
            print(v.shape)
        except:
            adj_matrix = nx.to_numpy_array(G)
            print("start computing eigen vectors")
            w, v = LA.eig(adj_matrix)
            indices = np.argsort(w)[::-1]
            v = v.transpose()[indices]
            # only save top 1000 eigenvectors
            np.save("citeseer/citeseer_eigenvector", v)
        
        # for j in range(0, 5):
        #     count = 0
        #     for i in range(v.shape[1]):
        #         if v[j, i].real < 1e-200 and v[j, i].real > 0:
        #             print("real part smaller than 1e-200", j, i, v[j, i])
        #             count += 1
        #     print(j, count)

        adj_matrix = nx.to_numpy_array(G)
        v = v.real
        print(v[1497, :])
        print(v)
        zeros = 3312-np.count_nonzero(v, axis=1)
        np.savetxt("citeseer_eigenvector_nonzeros.txt", zeros, fmt="%d")
        plt.bar(range(len(zeros)), zeros)
        plt.xlabel("node index")
        plt.ylabel("number of zeros in eigenvectors")
        plt.savefig('plot.png', dpi=300, bbox_inches='tight')

        plt.show()

        exit()
        feat_data = np.zeros((num_nodes, feature_dim))
        assert(feature_dim <= 1000)
        for i in range(num_nodes):
            for j in range(feature_dim):
                feat_data[i, j] = v[j, i]

    return feat_data, labels, adj_lists

def run_model(dataset, initializer, seed, epochs, classify="node", batch_size=128, feature_dim=100, identity_dim=50):
    # merge run_cora and run_pubmed
    num_nodes_map = {"cora": 2708, "pubmed": 19717, "citeseer": 3312, "usa-airport": 1190, "brazil-airport": 131, 'europe-airport': 399}
    num_classes_map = {"cora": 7, "pubmed": 3, "citeseer": 6, "usa-airport": 4, "brazil-airport": 4, "europe-airport": 4}
    # enc1_dim_map = {"cora": 128, "pubmed": 128, "citeseer": 128,  "brazil-airport": 32}
    enc2_dim_map = {"cora": 32, "pubmed": 128, "citeseer": 128, "usa-airport":128, "brazil-airport": 32, "europe-airport": 64}
    enc1_num_samples_map = {"cora": 5, "pubmed": 10, "citeseer": 5, "usa-airport": 5, "brazil-airport": 5, "europe-airport": 5}
    enc2_num_samples_map = {"cora": 5, "pubmed": 25, "citeseer": 5, "usa-airport": 5, "brazil-airport": 5, "europe-airport": 5}
    attribute_dim = {"cora": 1433, "pubmed": 500, "citeseer": 3703}

    np.random.seed(seed)
    random.seed(seed)
    feat_data = []
    labels = []
    adj_lists = []
    num_nodes = num_nodes_map[dataset]
    num_classes = num_classes_map[dataset]
    # enc1_dim = enc1_dim_map[dataset]
    enc2_dim = enc2_dim_map[dataset]

    if dataset == "cora":
        feat_data, labels, adj_lists = load_cora(feature_dim, initializer)
    elif dataset == "pubmed":
        feat_data, labels, adj_lists = load_pubmed(feature_dim, initializer)
    elif dataset == "citeseer":
        feat_data, labels, adj_lists = load_citeseer(feature_dim, initializer)
    elif dataset == "usa-airport":
        feat_data, labels, adj_lists = load_usa_airport(feature_dim, initializer)
    elif dataset == "brazil-airport":
        feat_data, labels, adj_lists = load_brazil_airport(feature_dim, initializer)
    elif dataset == "europe-airport":
        feat_data, labels, adj_lists = load_europe_airport(feature_dim, initializer)
        

    # feat_data, labels, adj_lists = load_data(dataset, feature_dim, initializer)
    # print(feat_data)
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
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, enc2_dim, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = enc1_num_samples_map[dataset]
    enc2.num_samples = enc2_num_samples_map[dataset]


    graphsage = SupervisedGraphSage(num_classes, enc2)
#    graphsage.cuda()
    test_end_idx_map = {
        "cora": 1000
    }
    val_end_idx_map = {
        "cora": 1500
    }
    rand_indices = np.random.permutation(num_nodes)
    test_end_idx = test_end_idx_map[dataset]
    val_end_idx = val_end_idx_map[dataset]
    test = rand_indices[:test_end_idx]
    val = rand_indices[test_end_idx:val_end_idx]
    train = list(rand_indices[val_end_idx:])
    train_num = len(train)

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.01)
    times = []

    for epoch in range(epochs):
        # print("Epoch:", epoch)
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
            # if (batch == 0):
            #     print("Batch", batch, "Loss:", loss.item())

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
        with open("cora/cora.content") as fp:
            for i,line in enumerate(fp):
                info = line.strip().split()
                feat_data[i,:] = list(map(float, info[1:-1]))
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
    feat_data, labels, adj_lists = load_cora(num_nodes, feature_dim, initializer)
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
    print("isolates:", list(nx.isolates(G)))
    exit()

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
            v = np.load("pubmed-data/pubmed_eigenvector.npy")
            print(v.shape)
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
