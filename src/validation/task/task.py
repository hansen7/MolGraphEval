import abc
import dataclasses
from typing import Dict
import torch
from torch.utils.data import DataLoader, Dataset
from config.validation_config import ValidationConfig
from logger import CombinedLogger


@dataclasses.dataclass
class Task(abc.ABC):
    config: ValidationConfig

    @abc.abstractmethod
    def run(self, model: torch.nn.Module, device: torch.device) -> Dict:
        pass


@dataclasses.dataclass
class TrainValTestTask(Task):
    config: ValidationConfig
    model: torch.nn.Module
    device: torch.device
    optimizer: torch.optim.Optimizer
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    criterion_type: str
    # logger: CombinedLogger

    @abc.abstractmethod
    def train(self) -> float:
        pass

    @abc.abstractmethod
    def _eval(self, loader: DataLoader) -> float:
        pass

    @abc.abstractmethod
    def eval_val_dataset(self) -> float:
        pass

    @abc.abstractmethod
    def eval_test_dataset(self) -> float:
        pass
