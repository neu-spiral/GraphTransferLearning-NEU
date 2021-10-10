"""
Module loads data and also creates synthetic data

@author: Andrey Gritsenko
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""

import numpy as np
from itertools import permutations


def PermuteGraph(Adj, labels=None, P=None):
    """
    Permute graph according to permutation matrix P

    Inputs:
        ADJ Adjacency matrix of a graph to be permuted
        P Permutation matrix, such that B = P^T * A * P, where A is an old adjacency matrix and B is a new one
        LABELS List of node labels

    Outputs:
        BSIM Adjacency matrix of a new graph after permutation
        BLABELS List of node labels after permutation
    """

    if P is None:
        P = np.eye(len(Adj), dtype=int)[np.random.permutation(len(Adj))]

    new_Adj = np.matmul(np.matmul(P.T, Adj), P)

    new_Labels = np.matmul(P.T, labels) if labels is not None else None
    return new_Adj, new_Labels, P


def ReverseEdges(A, pct=None):
    """
    Invert elements of graph's adjacency matrix with probability Prob

    Inputs:
        A Adjacency matrix of a graph to be permuted
        PCT Percent of inverted edges

    Outputs:
        B Adjacency matrix of a new graph after edge inversion
    """

    B = np.triu(A, 1)  # get upper triangular matrix from input A

    # remove edges
    ind = np.where(B == 1)  # find all edges
    # define number of edges to remove and add
    if pct is None or not (0 < pct < 1):
        n_edges = min(int(np.floor(len(A) / 2) + 1), len(ind[0]))
    else:
        n_edges = int(len(ind[0]) * pct) + 1
    randperm = np.random.permutation(len(ind[0]))[:n_edges]  # randomly choose edges to delete
    B[(ind[0][randperm], ind[1][randperm])] = 0  # delete selected edges

    # add edges
    B0 = np.triu(abs(B - 1), 1)
    ind = np.where(B0 == 1)  # find all absent edges
    randperm = np.random.permutation(len(ind[0]))[:n_edges]  # randomly choose edges to add
    B[(ind[0][randperm], ind[1][randperm])] = 1  # add selected edges
    return B + B.T


def RemoveEdges(A, n_edges=0):
    """
    Remove certain amount of edges from a graph
    """
    n_edges = int(n_edges)
    if n_edges < 1 or n_edges > A.size - len(A):
        n_edges = int(np.floor(len(A) / 2) + 1)

    B = np.triu(A, 1)
    ind = np.where(B == 1)
    randperm = np.random.permutation(len(ind[0]))[:min(n_edges, len(ind[0]))]
    B[(ind[0][randperm], ind[1][randperm])] = 0
    return B + B.T


def GetDoublyStochasticMatrix(Labels, P):
    n_labels = Labels.shape[1]
    if n_labels != 1:
        cluster_size = len(Labels) / n_labels
    else:
        cluster_size = 6 if len(Labels) == 120 else 4

    Block = np.eye(len(Labels))
    for l in range(n_labels):
        Block[cluster_size * l:cluster_size * (l + 1), cluster_size * l:cluster_size * (l + 1)] = 1 / cluster_size
    Pds = np.matmul(Block, P)
    return Pds


def getNodeSimilarity(A, mode='adjacency', n_walks=10, walk_length=10, window_size=5, p=.25, q=4, n_negative=5):
    """
    Wrapper of function computing similarity between node pairs

    Inputs:
        A Adjacency matrix of a graph
        MODE Similarity metric between nodes
        NODES List of nodes to be used (if one graph is split in training and test sets)
        FEATURES Features of graph nodes (used for some similarity metrics)

    Outputs:
        NODE_PAIRS List of node pairs, for which similarity was computed
        SIMILARITY Measure of similarity between two nodes
    """
    if mode == 'randomwalk':
        from random_walks import getRWSimilarity
        target, context, similarity, neg_samples = getRWSimilarity(A, n_walks, walk_length, window_size, p, q,
                                                                   n_negative)
    else:
        node_pairs = np.array(list(permutations(range(len(A)), 2)))
        np.random.shuffle(node_pairs)
        target = node_pairs[:, 0]
        context = node_pairs[:, 1]
        n_negative = 0
        neg_samples = None

        similarity = np.zeros((len(node_pairs), n_negative + 1))
        similarity[:, 0] = [A[i, j] for i, j in node_pairs]

    return target, context, similarity, neg_samples
