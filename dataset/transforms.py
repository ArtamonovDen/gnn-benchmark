import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import is_undirected, to_networkx
import numpy as np
import igraph as ig

class TransfromController:
    ...
    # TODO manage transfroms



# TODO: add class from NDD transform (like max degree transfrom)

class NDDTransform(BaseTransform):
    """
    Apply Node Distance Distribution matrix 
    to node features
    """
    def __init__(self, max_diameter, cat=True):
        self.max_deameter = max_diameter + 1
        self.cat = cat

    def __call__(self, data):

        x = data.x

        undirected = is_undirected(data)
        g = to_networkx(data, to_undirected=undirected)
        g = ig.Graph.from_networkx(g)

        num_nodes = g.vcount()
        bins = np.append(np.arange(0, self.max_deameter), float('inf'))

        mode_g = "ALL" if undirected else "OUT"
        sp = g.shortest_paths_dijkstra(mode=mode_g)

        distance_histogram = np.array([np.histogram(sp[v], bins=bins)[0] for v in range(0, num_nodes)])

        distrib_mat = distance_histogram / (num_nodes - 1)
        distrib_mat = torch.from_numpy(distrib_mat, dtype=torch.Float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, distrib_mat.to(x.dtype)], dim=-1)
        else:
            data.x = distrib_mat

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.max_degree})'