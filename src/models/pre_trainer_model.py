import torch
from config.training_config import TrainingConfig


class PreTrainerModel(torch.nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

    def get_embeddings(self, *argv):
        return self.gnn.get_embeddings(*argv)
