import gmatch4py as gm
import networkx as nx

from validation.task.metrics import GraphPairMetric


class GraphEditDistanceDataset(GraphPairMetric):
    """Compare the distance (minimal edits) between two graphs"""

    def __init__(self, representation_path, config):
        super(GraphEditDistanceDataset, self).__init__(representation_path, config)

    def graph_pair_metric(
        self, graph_nx_one: nx.Graph, graph_nx_two: nx.Graph
    ) -> float:
        ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs are equal to 1
        result = ged.compare([graph_nx_one, graph_nx_two], None)
        return result[0][1] / 100
