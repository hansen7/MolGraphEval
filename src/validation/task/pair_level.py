import networkx as nx
import numpy as np

from validation.task.metrics import NodePairMetric


class LinkPredictionDataset(NodePairMetric):
    """To predict whether two nodes are connected."""

    def __init__(self, representation_path, config, split, **kwargs):
        super(LinkPredictionDataset, self).__init__(
            representation_path, config, split, **kwargs
        )

    def node_pair_metric(self, graph_nx, start_node, end_node):
        return int(graph_nx.has_edge(start_node, end_node))


class KatzIndexDataset(NodePairMetric):
    """The most basic global overlap statistic.
    To compute the Katz index we simply count the number of paths
    of all lengths between a pair of nodes."""

    def __init__(self, representation_path, config, split, **kwargs):
        super(KatzIndexDataset, self).__init__(
            representation_path, config, split, **kwargs
        )

    def node_pair_metric(
        self, graph_nx: nx.Graph, start_node: int, end_node: int
    ) -> float:
        alpha = 0.3
        I = np.identity(len(graph_nx.nodes))
        katz_matrix = np.linalg.inv(I - nx.to_numpy_array(graph_nx) * alpha) - I
        return katz_matrix[start_node][end_node]


class JaccardCoefficientDataset(NodePairMetric):
    r"""Compute the Jaccard coefficient of all node pairs.

    Jaccard coefficient of nodes `u` and `v` is defined as
    .. math::
        \frac{|\Gamma(u) \cap \Gamma(v)|}{|\Gamma(u) \cup \Gamma(v)|}
    where $\Gamma(u)$ denotes the set of neighbors of $u$.

    References
    ----------
    .. [1] D. Liben-Nowell, J. Kleinberg.
           The Link Prediction Problem for Social Networks (2004).
           http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    """

    def __init__(self, representation_path, config, split, **kwargs):
        super(JaccardCoefficientDataset, self).__init__(
            representation_path, config, split, **kwargs
        )

    def node_pair_metric(self, graph_nx, start_node, end_node):
        preds = nx.jaccard_coefficient(graph_nx, [(start_node, end_node)])
        for u, v, p in preds:
            return p


# class GraphEditDistanceDataset(GraphPairMetric):
#     """ Compare the distance (minimal edits) between two graphs"""
#     def __init__(self, representation_path, config):
#         super(GraphEditDistanceDataset, self).__init__(
#             representation_path, config)
#
#     def graph_pair_metric(
#     self, graph_nx_one: nx.Graph, graph_nx_two: nx.Graph) -> float:
#         ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs equal to 1
#         result = ged.compare([graph_nx_one, graph_nx_two], None)
#         return result[0][1] / 100
