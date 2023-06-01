import networkx as nx
import numpy as np
from networkx.algorithms.distance_measures import diameter
from rdkit.Chem import Fragments

from datasets import graph_data_obj_to_mol_simple, graph_data_obj_to_nx_simple
from validation.task.metrics import GraphLevelMetric


class GraphDiameterDataset(GraphLevelMetric):
    """Calculate 2D graph diameter, i.e. longest path."""

    def __init__(self, representation_path, config, split, **kwargs):
        super(GraphDiameterDataset, self).__init__(
            representation_path, config, split, **kwargs
        )

    @staticmethod
    def graph_level_metric(graph, method="largest") -> int:
        graph_nx = graph_data_obj_to_nx_simple(graph)
        if nx.is_connected(graph_nx):
            return diameter(graph_nx)
        elif method == "largest":
            graphs = [
                graph_nx.subgraph(c).copy() for c in nx.connected_components(graph_nx)
            ]
            return max([diameter(g) for g in graphs])
        return None


class CycleBasisDataset(GraphLevelMetric):
    def __init__(self, representation_path, config, split, **kwargs):
        super(CycleBasisDataset, self).__init__(
            representation_path, config, split, **kwargs
        )

    @staticmethod
    def graph_level_metric(graph):
        """Calculate number of cycles."""
        graph_nx = graph_data_obj_to_nx_simple(graph)
        return len(nx.cycle_basis(graph_nx))


class AssortativityCoefficientDataset(GraphLevelMetric):
    """Compute degree assortativity of graph.

    Assortativity measures the similarity of connections
    in the graph with respect to the node degree."""

    def __init__(self, representation_path, config, split, **kwargs):
        super(AssortativityCoefficientDataset, self).__init__(
            representation_path, config, split, **kwargs
        )

    @staticmethod
    def graph_level_metric(graph) -> float:
        graph_nx = graph_data_obj_to_nx_simple(graph)
        ret = nx.algorithms.assortativity.degree_assortativity_coefficient(G=graph_nx)
        if np.isnan(ret):
            return None
        return ret


class AverageClusteringCoefficientDataset(GraphLevelMetric):
    """Estimates the average clustering coefficient of a graph.

    The local clustering of each node in `G` is the fraction of triangles
    that actually exist over all possible triangles in its neighborhood.
    The average clustering coefficient of a graph `G` is the mean of
    local clusters.

    This function finds an approximate average clustering coefficient
    for G by repeating `n` times (defined in `trials`) the following
    experiment: choose a node at random, choose two of its neighbors
    at random, and check if they are connected. The approximate
    coefficient is the fraction of triangles found over the number
    of trials [1]_."""

    def __init__(self, representation_path, config, split, **kwargs):
        super(AverageClusteringCoefficientDataset, self).__init__(
            representation_path, config, split, **kwargs
        )

    @staticmethod
    def graph_level_metric(graph) -> float:
        graph_nx = graph_data_obj_to_nx_simple(graph)
        return nx.algorithms.approximation.average_clustering(G=graph_nx)


class NodeConnectivityDataset(GraphLevelMetric):
    """Returns an approximation for node connectivity for a graph or digraph G.

    Node connectivity is equal to the minimum number of nodes that
    must be removed to disconnect G or render it trivial. By Menger's theorem,
    this is equal to the number of node independent paths (paths that
    share no nodes other than source and target).

    This algorithm is based on a fast approximation that gives an strict lower
    bound on the actual number of node independent paths between two nodes [1]_.
    It works for both directed and undirected graphs.

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf"""

    def __init__(self, representation_path, config, split, **kwargs):
        super(NodeConnectivityDataset, self).__init__(
            representation_path, config, split, **kwargs
        )

    @staticmethod
    def graph_level_metric(graph) -> int:
        graph_nx = graph_data_obj_to_nx_simple(graph)
        return nx.algorithms.approximation.node_connectivity(G=graph_nx)


# class BenzeneRingDataset(GraphLevelMetric):
#     """Calculate the number of Benzene rings."""
#
#     def __init__(self, representation_path, config, split, **kwargs):
#         super(BenzeneRingDataset, self).__init__(
#             representation_path, config, split, **kwargs)
#
#     @staticmethod
#     def graph_level_metric(graph) -> int:
#         mol = graph_data_obj_to_mol_simple(
#             graph.x, graph.edge_index, graph.edge_attr)
#         return Fragments.fr_benzene(mol)


RDKIT_fragments_valid = [
    "fr_epoxide",
    "fr_lactam",
    "fr_morpholine",
    "fr_oxazole",
    "fr_tetrazole",
    "fr_N_O",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_morpholine",
    "fr_piperdine",
    "fr_thiazole",
    "fr_thiophene",
    "fr_urea",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_azo",
    "fr_benzene",
    "fr_imidazole",
    "fr_imide",
    "fr_piperzine",
    "fr_pyridine",
]


class RDKiTFragmentDataset(GraphLevelMetric):
    """Calculate the number of Benzene rings."""

    def __init__(self, representation_path, config, split, des):
        super(RDKiTFragmentDataset, self).__init__(
            representation_path, config, split, des=des
        )

    # @staticmethod
    def graph_level_metric(self, graph):
        assert self.des in RDKIT_fragments_valid
        mol = graph_data_obj_to_mol_simple(graph.x, graph.edge_index, graph.edge_attr)
        cmd = compile("Fragments.%s(mol)" % self.des, "<string>", "eval")
        try:
            return eval(cmd)
        except:
            return


class DownstreamDataset(GraphLevelMetric):
    def __init__(self, representation_path, config, split, des):
        super(DownstreamDataset, self).__init__(
            representation_path, config, split, des=des
        )

    @staticmethod
    def graph_level_metric(graph):
        labels = graph.y.cpu().numpy().ravel()
        return labels
        # if ((labels ** 2) != 1).any():
        #     return None
        # return (labels + 1)/2
