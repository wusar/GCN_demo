import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import PPI
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
import torch
import scipy.sparse as sp
import numpy as np
import json


def encode_onehot(labels):
    classes = set(labels)  # 得到所有类别，利用set去重
    classes_dict = {
        c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
    }  # 为类别分配one-hot编码
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data_ppi(path="./datasets/ppi/", dataset="ppi"):
    """Load citation network dataset (cora only for now)"""
    print("Loading {} dataset...".format(dataset))

    idx_features_labels = np.load(path + "/ppi-feats.npy")
    print("feature of ppi: ", idx_features_labels, idx_features_labels.shape)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}  # 每篇论文的索引是多少

    edges_unordered = []
    with open(path + "/ppi-G.json") as f:
        json_format = json.load(f)
        for nodes in json_format["nodes"]:
            node_id = int(nodes["id"])

        for edges in json_format["links"]:
            source = edges["source"]
            target = edges["target"]
            if source != target:
                edges_unordered.append([source, target])
    edges_unordered = np.array(edges_unordered)
    print("edges of ppi: ", edges_unordered)
    print("edges of ppi: ", edges_unordered.shape)
    # 删除在edges_unordered中出现但是没有在idx_map中的引用记录
    edges_unordered = np.array(
        [edge for edge in edges_unordered if edge[0] in idx_map and edge[1] in idx_map]
    )

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())),  # flatten展开成一维向量
        dtype=np.dtype(int),
    ).reshape(
        edges_unordered.shape
    )  # 将id相对应的边，改成索引相对应的边。将edges_unordered.flatten()中的值，输入get函数中，返回value
    adj = sp.coo_matrix(
        (
            np.ones(edges.shape[0]),
            (edges[:, 0], edges[:, 1]),
        ),  # (edges[:, 0], edges[:, 1])这些位置的值为1
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)  # 对邻接矩阵采用均值归一化
    adj = normalize(adj + sp.eye(adj.shape[0]))

    num_nodes = adj.shape[0]  # 节点数
    idx_train = range(0, int(num_nodes * 0.5))
    idx_val = range(int(num_nodes * 0.5), int(num_nodes * 0.7))
    idx_test = range(int(num_nodes * 0.7), num_nodes)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data(path="./datasets/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print("Loading {} dataset...".format(dataset))
    if dataset == "ppi":
        return load_data_ppi(path, dataset)
    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str)
    )
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}  # 每篇论文的索引是多少
    edges_unordered = np.genfromtxt(
        "{}{}.cites".format(path, dataset), dtype=np.dtype(str)
    )

    # 删除在edges_unordered中出现但是没有在idx_map中的引用记录
    edges_unordered = np.array(
        [edge for edge in edges_unordered if edge[0] in idx_map and edge[1] in idx_map]
    )

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())),  # flatten展开成一维向量
        dtype=np.dtype(int),
    ).reshape(
        edges_unordered.shape
    )  # 将id相对应的边，改成索引相对应的边。将edges_unordered.flatten()中的值，输入get函数中，返回value
    adj = sp.coo_matrix(
        (
            np.ones(edges.shape[0]),
            (edges[:, 0], edges[:, 1]),
        ),  # (edges[:, 0], edges[:, 1])这些位置的值为1
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)  # 对邻接矩阵采用均值归一化
    adj = normalize(adj + sp.eye(adj.shape[0]))

    num_nodes = adj.shape[0]  # 节点数
    idx_train = range(0, int(num_nodes * 0.5))
    idx_val = range(int(num_nodes * 0.5), int(num_nodes * 0.7))
    idx_test = range(int(num_nodes * 0.7), num_nodes)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
