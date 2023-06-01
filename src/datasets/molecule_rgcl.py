import torch, numpy as np
from itertools import repeat
from torch_geometric.data import Data
from .molecule_datasets import MoleculeDataset

# from torch_geometric.utils import subgraph, to_networkx


def drop_nodes_prob(data, aug_ratio, node_score):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    node_prob = node_score.float()
    node_prob += 0.001
    node_prob = np.array(node_prob)
    node_prob /= node_prob.sum()

    idx_nondrop = np.random.choice(
        node_num, node_num - drop_num, replace=False, p=node_prob
    )
    idx_drop_set = set(np.setdiff1d(np.arange(node_num), idx_nondrop).tolist())
    idx_nondrop.sort()

    idx_dict = np.zeros((idx_nondrop[-1] + 1,), dtype=np.int64)
    idx_dict[idx_nondrop] = np.arange(len(idx_nondrop), dtype=np.int64)

    edge_index = data.edge_index.numpy()

    edge_mask = []
    for n in range(edge_num):
        if not (edge_index[0, n] in idx_drop_set or edge_index[1, n] in idx_drop_set):
            edge_mask.append(n)
    edge_mask = np.asarray(edge_mask, dtype=np.int64)
    edge_index = idx_dict[edge_index[:, edge_mask]]
    try:
        data.edge_index = torch.from_numpy(edge_index)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data

    return data


def drop_nodes_cp(data, aug_ratio, node_score):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    node_prob = max(node_score.float()) - node_score.float()
    node_prob += 0.001
    node_prob = np.array(node_prob)
    node_prob /= node_prob.sum()

    idx_nondrop = np.random.choice(
        node_num, node_num - drop_num, replace=False, p=node_prob
    )
    idx_drop_set = set(np.setdiff1d(np.arange(node_num), idx_nondrop).tolist())
    idx_nondrop.sort()

    idx_dict = np.zeros((idx_nondrop[-1] + 1,), dtype=np.int64)
    idx_dict[idx_nondrop] = np.arange(len(idx_nondrop), dtype=np.int64)

    edge_index = data.edge_index.numpy()

    edge_mask = []
    for n in range(edge_num):
        if not (edge_index[0, n] in idx_drop_set or edge_index[1, n] in idx_drop_set):
            edge_mask.append(n)
    edge_mask = np.asarray(edge_mask, dtype=np.int64)
    edge_index = idx_dict[edge_index[:, edge_mask]]
    try:
        data.edge_index = torch.from_numpy(edge_index)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data

    return data


def subgraph_prob(data, aug_ratio, node_score):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    node_prob = node_score.float()
    node_prob += 0.001
    node_prob = np.array(node_prob)
    node_prob /= node_prob.sum()

    neighbors = {i: [] for i in range(node_num + 1)}
    edge_index_list = edge_index.T.tolist()
    for i, j in edge_index_list:
        neighbors[i].append(j)

    root = np.random.choice(node_num, 1, p=node_prob)[0]
    idx_sub = {
        root,
    }
    idx_neigh = set(neighbors[root]).difference([root])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break

        list_neigh = list(idx_neigh)
        list_neigh.sort()

        neigh_prob = node_prob[np.array(list_neigh)]
        neigh_prob /= neigh_prob.sum()

        sample_node = np.random.choice(list_neigh, 1, p=neigh_prob)[0]
        idx_sub.add(sample_node)
        idx_neigh.union(neighbors[sample_node])
        idx_neigh.difference_update(idx_sub)

    idx_nondrop = list(idx_sub)
    idx_nondrop.sort()

    idx_dict = np.zeros((idx_nondrop[-1] + 1,), dtype=np.int64)
    idx_dict[idx_nondrop] = np.arange(len(idx_nondrop), dtype=np.int64)

    edge_mask = []
    for n in range(edge_num):
        if edge_index[0, n] in idx_sub and edge_index[1, n] in idx_sub:
            edge_mask.append(n)
    edge_mask = np.asarray(edge_mask, dtype=np.int64)
    edge_index = idx_dict[edge_index[:, edge_mask]]
    try:
        data.edge_index = torch.from_numpy(edge_index)  # .transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data
    return data


def subgraph_cp(data, aug_ratio, node_score):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    node_prob = max(node_score.float()) - node_score.float()
    node_prob += 0.001
    node_prob = np.array(node_prob)
    node_prob /= node_prob.sum()

    neighbors = {i: [] for i in range(node_num + 1)}
    edge_index_list = edge_index.T.tolist()
    for i, j in edge_index_list:
        neighbors[i].append(j)

    root = np.random.choice(node_num, 1, p=node_prob)[0]
    idx_sub = {
        root,
    }
    idx_neigh = set(neighbors[root]).difference([root])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break

        list_neigh = list(idx_neigh)
        list_neigh.sort()

        neigh_prob = node_prob[np.array(list_neigh)]
        neigh_prob /= neigh_prob.sum()

        sample_node = np.random.choice(list_neigh, 1, p=neigh_prob)[0]
        idx_sub.add(sample_node)
        idx_neigh.union(neighbors[sample_node])
        idx_neigh.difference_update(idx_sub)

    idx_nondrop = list(idx_sub)
    idx_nondrop.sort()

    idx_dict = np.zeros((idx_nondrop[-1] + 1,), dtype=np.int64)
    idx_dict[idx_nondrop] = np.arange(len(idx_nondrop), dtype=np.int64)

    edge_mask = []
    for n in range(edge_num):
        if edge_index[0, n] in idx_sub and edge_index[1, n] in idx_sub:
            edge_mask.append(n)
    edge_mask = np.asarray(edge_mask, dtype=np.int64)
    edge_index = idx_dict[edge_index[:, edge_mask]]
    try:
        data.edge_index = torch.from_numpy(edge_index)  # .transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data
    return data


class MoleculeDataset_RGCL(MoleculeDataset):
    # used in RGCL
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset=None,
        empty=False,
        aug="none",
        aug_ratio=None,
    ):

        self.aug = aug
        self.aug_ratio = aug_ratio
        # self.augmentations = [
        #     self.node_drop,
        #     self.subgraph,
        #     self.edge_pert,
        #     self.attr_mask,
        #     lambda x: x,
        # ]
        super(MoleculeDataset_RGCL, self).__init__(
            root, transform, pre_transform, pre_filter, dataset, empty
        )
        self.transform, self.pre_transform, self.pre_filter = (
            transform,
            pre_transform,
            pre_filter,
        )

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

        self.node_score = torch.zeros([self.data["x"].size(0)], dtype=torch.half)

    def set_augMode(self, aug_mode):
        self.aug_mode = aug_mode

    def set_augStrength(self, aug_strength):
        self.aug_strength = aug_strength

    def set_augProb(self, aug_prob):
        self.aug_prob = aug_prob

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        if self.aug == "dropN":
            nodes_score = self.node_score[
                self.slices["x"][idx] : self.slices["x"][idx + 1]
            ]
            data = drop_nodes_prob(data, self.aug_ratio, nodes_score)
        elif self.aug == "dropN_cp":
            nodes_score = self.node_score[
                self.slices["x"][idx] : self.slices["x"][idx + 1]
            ]
            data = drop_nodes_cp(data, self.aug_ratio, nodes_score)
        elif self.aug == "none":
            None
        else:
            print("augmentation error")
            assert False

        return data
