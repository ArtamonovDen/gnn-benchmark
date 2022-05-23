import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import is_undirected, to_networkx
import numpy as np
import torch_geometric.transforms as T
import igraph as ig


class TransformController:

    __supported_types = ("ndd", "deg", "none")

    @classmethod
    def get_transform(cls, transform, max_diam, max_degree, cat=True):
        if transform not in cls.__supported_types:
            raise ValueError(f"Transform of type {transform} is not supported")

        if transform == "ndd":
            return NDDTransform(max_diameter=max_diam, cat=cat)
        elif transform == "deg":
            return T.OneHotDegree(max_degree=max_degree, cat=cat)
        else:
            return None

    @classmethod
    def get_supported_types(cls):
        return cls.__supported_types


class NDDTransform(BaseTransform):
    """
    Apply Node Distance Distribution matrix 
    to node features
    """
    def __init__(self, max_diameter, cat=True):
        self.max_diameter = max_diameter + 1
        self.cat = cat

    def __call__(self, data):

        x = data.x

        undirected = is_undirected(data.edge_index)
        g = to_networkx(data, to_undirected=undirected)
        g = ig.Graph.from_networkx(g)

        num_nodes = g.vcount()
        bins = np.append(np.arange(0, self.max_diameter), float('inf'))

        mode_g = "ALL" if undirected else "OUT"
        sp = g.shortest_paths_dijkstra(mode=mode_g)

        distance_histogram = np.array([np.histogram(sp[v], bins=bins)[0] for v in range(0, num_nodes)])

        distrib_mat = distance_histogram / (num_nodes - 1)
        distrib_mat = torch.from_numpy(distrib_mat).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, distrib_mat.to(x.dtype)], dim=-1)
        else:
            data.x = distrib_mat

        return data

    def __repr__(self) -> str:
        return "{self.__class__.__name__}({self.max_diameter})"
