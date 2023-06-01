from torch.utils.data import Dataset


class ProberDataset(Dataset):
    """A dataset class that holds (representation, label) pairs."""

    def __init__(self, representations, labels):
        assert len(representations) == len(labels)
        self.input_dim = len(representations[0])  # 300
        self.representations = representations
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "representation": self.representations[idx],
            "label": self.labels[idx],
        }
