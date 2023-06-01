import pickle as pkl

import numpy as np
from sklearn.metrics import pairwise_distances

from validation.utils import get_dataset_extraction, get_dataset_split, get_smiles_list


class MeanAverageDistance:
    """Calculate the mean cosine distance of node representations.
    Ref: https://arxiv.org/pdf/1909.03211.pdf)."""

    def __init__(self, representation_path, config, split):
        self.representation_path = representation_path
        with open(representation_path, "rb") as f:
            (
                self.dataset,
                self.graph_repr_list,
                self.node_repr_list,
                self.smiles,
            ) = pkl.load(f)

        dataset = get_dataset_extraction(config=config)
        smiles_list = get_smiles_list(config=config)
        dataset_splits = get_dataset_split(
            config=config, dataset=dataset, smiles_list=smiles_list
        )
        # "train", "validation", "test"
        self.dataset = dataset_splits[split]

    def calculate_metric(self):
        mads = []
        for node_repr in self.node_repr_list:
            distance_mat = pairwise_distances(node_repr, metric="cosine")
            mad = np.mean(distance_mat)
            mads.append(mad)
        print("{} MAD value is {}".format(self.representation_path, np.mean(mads)))
