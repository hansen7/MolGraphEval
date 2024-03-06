# Ref: {GitHub}/lsh0520/RGCL/transferLearning/chem/pretrain_rgcl.py
# Ref: https://arxiv.org/abs/2010.13902
# TODO: TO UPDATE
""" GRAPH SSL Pre-Training via InfoGraph [InfoGraph]
i.e., maps nodes in similar structural contexts to closer embeddings
Ref Paper: Sec. 5.2 and Appendix G of
            https://arxiv.org/abs/1905.12265 ;
           which is adapted from
            https://arxiv.org/abs/1809.10341 ;
Ref Code: ${GitHub_Repo}/chem/pretrain_deepgraphinfomax.py """

import torch

from config.training_config import TrainingConfig
from pretrainers.pretrainer import PreTrainer
from models.graphmae import GraphMAEModel
from torch.utils.data import DataLoader
from logger import CombinedLogger
from util import get_lr
from typing import List


class GraphMAEPreTrainer(PreTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        model: GraphMAEModel,
        optimizer: List,
        device: torch.device,
        logger: CombinedLogger,
    ):
        super(GraphMAEPreTrainer, self).__init__(
            config=config,
            model=model,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )
        assert len(self.optimizer) == 2

    def train_for_one_epoch(self, train_data_loader: DataLoader) -> float:
        train_loss_accum = 0
        self.model.train()
        self.logger.train(num_batches=len(train_data_loader))

        for step, batch in enumerate(train_data_loader):
            batch = batch.to(self.device)
            self.optimizer[0].zero_grad()
            self.optimizer[1].zero_grad()
            pred_node = self.model.forward(batch)

            loss = self.model.loss(pred_node, batch)
            loss.backward()
            self.optimizer[0].step()
            self.optimizer[1].step()

            loss_float = float(loss.detach().cpu().item())
            train_loss_accum += loss_float
            self.logger(loss_float, 0.0, batch.num_graphs, get_lr(self.optimizer[0]))

        return train_loss_accum / (step + 1)

    def validate_model(self, val_data_loader) -> float:
        # self.logger.eval(num_batches=len(val_data_loader))
        self.logger.eval(num_batches=1)
        self.logger(0.0, 0.0, 1)
        return 0.0
