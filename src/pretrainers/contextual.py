""" GROVER, Contextual Property Prediction
Ref Paper: https://arxiv.org/abs/2007.02835
Ref Code: https://github.com/tencent-ailab/grover """

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.training_config import TrainingConfig
from logger import CombinedLogger
from models.contextual import ContextualModel
from pretrainers.pretrainer import PreTrainer
from util import get_lr


class ContextualPreTrainer(PreTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        model: ContextualModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: CombinedLogger,
    ):
        super(ContextualPreTrainer, self).__init__(
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
            node_pred = self.model.forward_cl(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            node_target = batch.atom_vocab_label
            loss = self.model.loss_cl(node_pred, node_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_float = float(loss.detach().cpu().item())
            train_loss_accum += loss_float
            # loss, accuracy, batch_size, learning_rate
            self.logger(loss_float, 0.0, batch.num_graphs, get_lr(self.optimizer))

        return train_loss_accum / (step + 1)

    # TODO
    def validate_model(self, val_data_loader: DataLoader) -> float:
        # self.logger.eval(num_batches=len(val_data_loader))
        self.logger.eval(num_batches=1)
        self.logger(0.0, 0.0, 1)
        return 0.0
