from typing import List

import networkx as nx

from datasets import graph_data_obj_to_mol_simple, graph_data_obj_to_nx_simple
from validation.task.metrics import NodeLevelMetric


class NodeCentralityDataset(NodeLevelMetric):
    """Calculate node centrality for each node in a graph."""

    def __init__(self, representation_path, config, split, **kwargs):
        super(NodeCentralityDataset, self).__init__(
            representation_path, config, split, **kwargs
        )

    @staticmethod
    def node_level_metric(graph):
        graph_nx = graph_data_obj_to_nx_simple(graph)
        try:
            centrality = nx.eigenvector_centrality(graph_nx, max_iter=1500)
        except nx.exception.PowerIterationFailedConvergence:
            print("PowerIterationFailedConvergence")
            return [0] * len(graph_nx.nodes())

        # centrality = nx.eigenvector_centrality_numpy(graph_nx)
        node_centrality = sorted(list(centrality.items()))

        return [c for v, c in node_centrality]


class NodeDegreeDataset(NodeLevelMetric):
    """Calculate node degrees for each node in a graph."""

    def __init__(self, representation_path, config, split, **kwargs):
        super(NodeDegreeDataset, self).__init__(
            representation_path, config, split, **kwargs
        )

    @staticmethod
    def node_level_metric(graph) -> List[int]:
        metrics = []
        mol = graph_data_obj_to_mol_simple(graph.x, graph.edge_index, graph.edge_attr)
        for _, atom in enumerate(mol.GetAtoms()):
            metrics.append(atom.GetDegree())
            # if atom.GetDegree() > 4:
            #     print(atom.GetDegree())
        return metrics


class NodeClusteringDataset(NodeLevelMetric):
    """Calculate clustering coefficient for each node in a graph."""

    def __init__(self, representation_path, config, split, **kwargs):
        super(NodeClusteringDataset, self).__init__(
            representation_path, config, split, **kwargs
        )

    @staticmethod
    def node_level_metric(graph) -> List[float]:
        graph_nx = graph_data_obj_to_nx_simple(graph)
        clustering_coefficient = nx.clustering(graph_nx)
        node_clustering = sorted(list(clustering_coefficient.items()))
        return [c for v, c in node_clustering]
