import torch
import igraph as ig
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_edges_from_igraph(iG: ig.Graph, is_directed: bool) -> torch.LongTensor:
    """
    Returns Pytorch Geometric edge representation as torch.tensor of
    pairs of node
    """
    print("Obtaining edge matrix")
    e_num = iG.ecount()
    coo_format_edges = list(zip(*iG.get_edgelist()))

    if not is_directed:
        # add back-edges manually for undirected graph
        source_nodes, sink_nodes = coo_format_edges[0], coo_format_edges[1]
        coo_format_edges[0] += sink_nodes
        coo_format_edges[1] += source_nodes
    pyg_edge_index = torch.tensor(coo_format_edges, dtype=torch.long)
    print(f"Edge coo matrix has shape {pyg_edge_index.shape}")

    expected_edge_num = e_num if is_directed else e_num * 2
    assert pyg_edge_index.shape == (2, expected_edge_num)
    return pyg_edge_index


def get_edges_with_weghts_from_igraph(iG: ig.Graph):
    """
    Returns Pytorch Geometric edge representation as torch.tensor of
    pairs of node and list of weights corresponding to each node
    """
    edges, edges_weights = [], []
    for e in iG.es:
        source, target = e.source, e.target
        weight = e["weight"]
        edges.append((source, target))
        edges_weights.append(weight)  # store twice for each directed edge

    coo_format_edges = list(zip(*edges))
    edge_index = torch.tensor(coo_format_edges, dtype=torch.long)
    edges_weights = torch.tensor(edges_weights, dtype=torch.float).reshape((1, -1))
    # TODO: edge weights to tensor
    print(f"Edge coo matrix has shape {edge_index.shape}")
    print(f"Sample of edge weights: {edges_weights[:10]}")
    return edge_index, edges_weights


def get_degree_matrix(iG: ig.Graph) -> torch.FloatTensor:
    """
    Return one-hot encoded node degree features fore each node
    as torch.tensor
    """
    degree_encoder = OneHotEncoder()
    d = np.array(iG.degree())
    d = d.reshape((len(d), 1))
    one_hot_d = degree_encoder.fit_transform(d)
    d = torch.FloatTensor(one_hot_d.toarray())
    print(f"Use one hot encoded degree matrix as X. Shape is {d.shape}")
    return d


def get_adj_matrix(iG: ig.Graph) -> torch.FloatTensor:
    """
    Returns adjacensy matrix as torch.tensor
    """
    # TODO: add edj weights and use Laplasian
    adj = iG.get_adjacency().data
    adj = torch.tensor(adj, dtype=torch.float64)
    print(f"Use adj matrix as X. Shape is {adj.shape}")
    return adj
