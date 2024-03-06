import torch
from torch.utils.data import DataLoader

from config.training_config import TrainingConfig
from logger import CombinedLogger
from models.graphmvp import GraphMVPModel
from pretrainers.pretrainer import PreTrainer
from util import get_lr


class GraphMVPPreTrainer(PreTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        model: GraphMVPModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: CombinedLogger,
    ):
        super(GraphMVPPreTrainer, self).__init__(
            config=config,
            model=model,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )

    def train_for_one_epoch(self, train_data_loader: DataLoader) -> float:
        self.model.train()
        train_loss_accum = 0.0
        self.logger.train(num_batches=len(train_data_loader))

        for step, batch in enumerate(train_data_loader):
            batch = batch.to(self.device)
            repr_2d, repr_3d = self.model(batch)
            loss = self.model.loss(repr_2d, repr_3d)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_float = float(loss.detach().cpu().item())
            train_loss_accum += loss_float
            self.logger(loss_float, 0.0, batch.num_graphs, get_lr(self.optimizer))

        return train_loss_accum / (step + 1)

    def validate_model(self, val_data_loader: DataLoader) -> float:
        return 0
