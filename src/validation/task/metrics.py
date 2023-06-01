# import pdb
import pickle as pkl
import random
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from datasets import graph_data_obj_to_nx_simple
from validation.dataset import ProberDataset
from validation.utils import get_dataset_extraction, get_dataset_split, get_smiles_list


class NodeLevelMetric(ABC):
    def __init__(self, representation_path, config, split, **kwargs):
        self.representation_path = representation_path
        self.__dict__.update(kwargs)
        self.config = config
        with open(representation_path, "rb") as f:
            (
                _,
                self.node_repr_list,
                self.smiles,
            ) = pkl.load(f)
        self.num_graph = len(self.smiles)

        dataset = get_dataset_extraction(config=config)
        smiles_list = get_smiles_list(config=config)
        dataset_splits = get_dataset_split(
            config=config, dataset=dataset, smiles_list=smiles_list
        )
        # "train", "val", "test"
        self.dataset = dataset_splits[split]

    def create_datasets(self):
        labels = []
        representations = []
        for graph_idx in range(self.num_graph):
            graph = self.dataset[graph_idx]
            node_repr = self.node_repr_list[graph_idx]
            metrics = self.node_level_metric(graph)
            assert len(metrics) == len(node_repr)
            for node_idx in range(len(metrics)):
                labels.append(metrics[node_idx])
                representations.append(node_repr[node_idx])
        return ProberDataset(representations, labels)

    @abstractmethod
    def node_level_metric(self):
        pass


class GraphLevelMetric(ABC):
    def __init__(self, representation_path, config, split, **kwargs):
        self.representation_path = representation_path
        self.__dict__.update(kwargs)
        self.config = config
        with open(representation_path, "rb") as f:
            (self.graph_repr_list, _, self.smiles) = pkl.load(f)
        self.num_graph = len(self.smiles)
        dataset = get_dataset_extraction(config=config)
        smiles_list = get_smiles_list(config=config)
        dataset_splits = get_dataset_split(
            config=config, dataset=dataset, smiles_list=smiles_list
        )
        # "train", "validation", "test"
        self.dataset = dataset_splits[split]

    def create_datasets(self):
        labels = []
        representations = []
        for graph_idx in range(self.num_graph):
            graph = self.dataset[graph_idx]
            graph_repr = self.graph_repr_list[graph_idx]
            metrics = self.graph_level_metric(graph)
            if metrics is not None:
                labels.append(metrics)
                representations.append(graph_repr)
        return ProberDataset(representations, labels)

    @abstractmethod
    def graph_level_metric(self):
        pass


class NodePairMetric(ABC):
    def __init__(self, representation_path, config, split, **kwargs):
        self.representation_path = representation_path
        self.__dict__.update(kwargs)
        self.config = config
        with open(representation_path, "rb") as f:
            (
                _,
                self.node_repr_list,
                self.smiles,
            ) = pkl.load(f)
        self.num_graph = len(self.smiles)

        dataset = get_dataset_extraction(config=config)
        smiles_list = get_smiles_list(config=config)
        dataset_splits = get_dataset_split(
            config=config, dataset=dataset, smiles_list=smiles_list
        )
        # "train", "validation", "test"
        self.dataset = dataset_splits[split]

    def create_datasets(self, num_pairs=30):
        labels = []
        representations = []
        for graph_idx in range(self.num_graph):
            graph = self.dataset[graph_idx]
            node_repr = self.node_repr_list[graph_idx]
            num_nodes = len(node_repr)
            graph_nx = graph_data_obj_to_nx_simple(graph)
            for _ in range(num_pairs):
                start_node = random.choice(list(range(num_nodes)))
                end_node = random.choice(list(range(num_nodes)))
                if start_node != end_node:
                    representations.append(
                        np.concatenate(
                            [
                                node_repr[start_node],
                                node_repr[end_node],
                                np.multiply(node_repr[start_node], node_repr[end_node]),
                            ]
                        )
                    )
                    labels.append(self.node_pair_metric(graph_nx, start_node, end_node))
        return ProberDataset(representations, labels)

    @abstractmethod
    def node_pair_metric(self, graph_nx, start_node, end_node):
        pass


class GraphPairMetric(ABC):
    def __init__(self, representation_path, config):
        self.representation_path = representation_path
        self.config = config
        with open(representation_path, "rb") as f:
            (
                self.dataset,
                self.graph_repr_list,
                self.node_repr_list,
                self.smiles,
            ) = pkl.load(f)
        self.num_graph = len(self.smiles)

    def create_datasets(self, num_pairs=10000):
        def take_out_graph(dataset, idx):
            graph = dataset[idx]
            if self.config.pretrainer == "GraphCL" and isinstance(graph, tuple):
                graph = graph[0]  # remove the contrastive augmented data.
            graph_nx = graph_data_obj_to_nx_simple(graph)
            return graph_nx

        labels = []
        representations = []
        for _ in tqdm(range(num_pairs)):
            graph_one_idx = random.choice(list(range(self.num_graph)))
            graph_two_idx = random.choice(list(range(self.num_graph)))
            if graph_one_idx != graph_two_idx:
                graph_nx_one = take_out_graph(self.dataset, graph_one_idx)
                graph_nx_two = take_out_graph(self.dataset, graph_two_idx)
                graph_repr_one = self.graph_repr_list[graph_one_idx]
                graph_repr_two = self.graph_repr_list[graph_two_idx]

                representations.append(
                    np.concatenate(
                        [
                            graph_repr_one,
                            graph_repr_two,
                            np.multiply(graph_repr_one, graph_repr_two),
                        ]
                    )
                )
                labels.append(self.graph_pair_metric(graph_nx_one, graph_nx_two))
        return ProberDataset(representations, labels)

    @abstractmethod
    def graph_pair_metric(self, graph_nx_one, graph_nx_two):
        pass


class FineTune_Metric(ABC):
    def __init__(self, representation_path, config, split, **kwargs):
        self.representation_path = representation_path
        self.__dict__.update(kwargs)
        self.config = config
        with open(representation_path, "rb") as f:
            _, _, self.smiles = pkl.load(f)
        self.num_graph = len(self.smiles)

        dataset = get_dataset_extraction(config=config)
        smiles_list = get_smiles_list(config=config)
        dataset_splits = get_dataset_split(
            config=config, dataset=dataset, smiles_list=smiles_list
        )
        # "train", "val", "test"
        self.dataset = dataset_splits[split]

    def create_datasets(self):
        labels = []
        representations = []
        for graph_idx in range(self.num_graph):
            graph = self.dataset[graph_idx]
            # node_repr = self.node_repr_list[graph_idx]
            metrics = self.node_level_metric(graph)
            # assert len(metrics) == len(node_repr)
            for node_idx in range(len(metrics)):
                labels.append(metrics[node_idx])
                representations.append(graph)
        return ProberDataset(representations, labels)

    @abstractmethod
    def node_level_metric(self):
        pass
