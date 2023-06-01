import abc, torch, dataclasses

from typing import Union, List
from logger import CombinedLogger
from torch.utils.data import DataLoader, Dataset
from config.training_config import TrainingConfig


@dataclasses.dataclass
class PreTrainer(abc.ABC):
    config: TrainingConfig
    model: torch.nn.Module
    # optimizer: torch.optim.Optimizer
    optimizer: Union[torch.optim.Optimizer, List]
    device: torch.device
    logger: CombinedLogger

    @abc.abstractmethod
    def train_for_one_epoch(self, train_data_loader: Union[DataLoader, Dataset]):
        pass

    @abc.abstractmethod
    def validate_model(self, val_data_loader: Union[DataLoader, Dataset]):
        pass
