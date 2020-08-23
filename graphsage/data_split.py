import numpy as np

label_map = {}
label_node_list_map = {}

num_nodes = 2708
num_feats = 1433

num_classes = 7

feat_data = np.zeros((num_classes * 20, num_feats))
labels = np.empty((num_nodes,1), dtype=np.int64)
with open("cora/cora.content") as fp:
    for i,line in enumerate(fp):
        info = line.strip().split()
        feat_data[i,:] = list(map(float, info[1:-1]))
        node_map[info[0]] = i
        if not info[-1] in label_map:
            label_map[info[-1]] = len(label_map)
            label_node_list_map[len(label_map)] = []
        labels[i] = label_map[info[-1]]
        label_node_list_map[labels[i]].append(i)

    train_idx = []
    for label, node_list in label_node_list_map.items():
        train_idx.extend(np.random.choice(node_list, 20))