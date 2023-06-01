""" GPT-GNN pre-training
Current version only supports the node reconstruction for molecular data.
Because the current GIN model only supports node representation.
In the future, we will add edge reconstruction and more general graph data."""

import torch
from torch.utils.data import DataLoader

from config.training_config import TrainingConfig
from logger import CombinedLogger
from models import GPTGNNModel
from pretrainers.pretrainer import PreTrainer
from util import get_lr


class GPTGNNPreTrainer(PreTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        model: GPTGNNModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: CombinedLogger,
    ) -> None:
        super(GPTGNNPreTrainer, self).__init__(
            config=config,
            model=model,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )

    def train_for_one_epoch(self, train_data_loader: DataLoader) -> float:
        self.model.train()
        self.logger.train(num_batches=len(train_data_loader))
        gpt_loss_accum = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        for step, batch in enumerate(train_data_loader):
            batch = batch.to(self.device)

            node_pred = self.model(batch)
            target = batch.next_x

            loss = criterion(node_pred.double(), target)
            acc = compute_accuracy(node_pred, target)

            loss_float = loss.detach().cpu().item()
            gpt_loss_accum += loss.detach().cpu().item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.logger(loss_float, acc, batch.num_graphs, get_lr(self.optimizer))

        return gpt_loss_accum / (step + 1)

    # TODO
    def validate_model(self, val_data_loader: DataLoader) -> float:
        # self.logger.eval(num_batches=len(val_data_loader))
        self.logger.eval(num_batches=1)
        self.logger(0.0, 0.0, 1)
        return 0.0


def compute_accuracy(pred, target):
    return float(
        torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()
    ) / len(pred)
