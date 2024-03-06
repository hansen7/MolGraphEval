""" GRAPH SSL Pre-Training via Edge Prediction
Ref Paper: Sec. 5.2 and Appendix G of
            https://arxiv.org/abs/1905.12265 ;
           which is adapted from
            https://arxiv.org/abs/1706.02216 ;

Ref Code: ${GitHub_Repo}/chem/pretrain_edgepred.py """


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.training_config import TrainingConfig
from logger import CombinedLogger
from models.edge_prediction import EdgePredictionModel
from pretrainers.pretrainer import PreTrainer
from util import get_lr


def do_EdgePred(node_repr, batch, criterion):
    # positive/negative scores -> inner product of node features
    positive_score = torch.sum(
        node_repr[batch.edge_index[0, ::2]] * node_repr[batch.edge_index[1, ::2]], dim=1
    )
    negative_score = torch.sum(
        node_repr[batch.negative_edge_index[0]]
        * node_repr[batch.negative_edge_index[1]],
        dim=1,
    )

    edgepred_loss = criterion(
        positive_score, torch.ones_like(positive_score)
    ) + criterion(negative_score, torch.zeros_like(negative_score))
    edgepred_acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(
        torch.float32
    ) / float(2 * len(positive_score))
    edgepred_acc = edgepred_acc.detach().cpu().item()

    return edgepred_loss, edgepred_acc


class EdgePredictionPreTrainer(PreTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        model: EdgePredictionModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: CombinedLogger,
    ) -> None:
        super(EdgePredictionPreTrainer, self).__init__(
            config=config,
            model=model,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )

    def train_for_one_epoch(self, train_data_loader: DataLoader) -> float:
        self.model.train()
        self.logger.train(num_batches=len(train_data_loader))

        loss_accum = 0.0

        for step, batch in enumerate(train_data_loader):
            batch = batch.to(self.device)
            node_repr = self.model(batch)
            edgepred_loss, edgepred_acc = do_EdgePred(
                node_repr=node_repr, batch=batch, criterion=nn.BCEWithLogitsLoss()
            )
            loss_float = edgepred_loss.detach().cpu().item()
            loss_accum += loss_float
            self.optimizer.zero_grad()
            edgepred_loss.backward()
            self.optimizer.step()
            self.logger(
                loss_float, edgepred_acc, batch.num_graphs, get_lr(self.optimizer)
            )

        return loss_accum / (step + 1)

    # TODO
    def validate_model(self, val_data_loader: DataLoader) -> float:
        # self.logger.eval(num_batches=len(val_data_loader))
        self.logger.eval(num_batches=1)
        self.logger(0.0, 0.0, 1)
        return 0.0


if __name__ == "__main__":
    pass
