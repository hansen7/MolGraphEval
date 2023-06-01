# Ref: {GitHub}/transferLearning_MoleculeNet_PPI
# Ref: https://arxiv.org/abs/2010.13902

import torch
from torch.utils.data import DataLoader
from config.training_config import TrainingConfig
from logger import CombinedLogger
from models.graph_cl import GraphCLModel
from pretrainers.pretrainer import PreTrainer
from util import get_lr


class GraphCLPreTrainer(PreTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        model: GraphCLModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: CombinedLogger,
    ):
        super(GraphCLPreTrainer, self).__init__(
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

        for step, (_, batch1, batch2) in enumerate(train_data_loader):
            batch1 = batch1.to(self.device)
            batch2 = batch2.to(self.device)

            x1 = self.model.forward_cl(
                batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch
            )
            x2 = self.model.forward_cl(
                batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch
            )
            loss = self.model.loss_cl(x1, x2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_float = float(loss.detach().cpu().item())
            train_loss_accum += loss_float
            self.logger(loss_float, 0.0, batch1.num_graphs, get_lr(self.optimizer))

        return train_loss_accum / (step + 1)

    # TODO
    def validate_model(self, val_data_loader: DataLoader) -> float:
        # self.logger.eval(num_batches=len(val_data_loader))
        self.logger.eval(num_batches=1)
        self.logger(0.0, 0.0, 1)
        return 0.0
