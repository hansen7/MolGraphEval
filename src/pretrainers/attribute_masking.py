""" GRAPH SSL Pre-Training via Attribute Masking
i.e., maps nodes in similar structural contexts to closer embeddings
Ref Paper: Sec. 3.1.2 and Appendix G of
            https://arxiv.org/abs/1905.12265 ;
Ref Code: ${GitHub_Repo}/chem/pretrain_contextpred.py """

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from config.training_config import TrainingConfig
from logger import CombinedLogger
from models.attribute_masking import AttributeMaskingModel
from pretrainers.pretrainer import PreTrainer
from util import get_lr


class AttributeMaskingPreTrainer(PreTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        model: AttributeMaskingModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: CombinedLogger,
    ) -> None:
        super(AttributeMaskingPreTrainer, self).__init__(
            config=config,
            model=model,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )

    def train_for_one_epoch(self, train_data_loader: DataLoader) -> float:
        self.model.train()
        self.logger.train(num_batches=len(train_data_loader))

        attributemask_loss_accum = 0.0

        for step, batch in enumerate(train_data_loader):
            batch = batch.to(self.device)

            loss, acc = do_AttrMasking(
                batch=batch,
                criterion=torch.nn.CrossEntropyLoss(),
                model=self.model,
            )
            loss_float = loss.detach().cpu().item()
            attributemask_loss_accum += loss_float
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.logger(loss_float, acc, batch.num_graphs, get_lr(self.optimizer))

        return attributemask_loss_accum / (step + 1)

    # TODO
    def validate_model(self, val_data_loader: DataLoader) -> float:
        # self.logger.eval(num_batches=len(val_data_loader))
        self.logger.eval(num_batches=1)
        self.logger(0.0, 0.0, 1)
        return 0.0


def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_cls = torch.max(pred.detach(), dim=1)[1]
    return float(torch.sum(pred_cls == target).cpu().item()) / len(pred)


def do_AttrMasking(
    batch: Batch, criterion: torch.nn.Module, model: AttributeMaskingModel
) -> Tuple[torch.Tensor, float]:
    target = batch.mask_node_label[:, 0]
    node_pred = model.forward(batch)
    attributemask_loss = criterion(node_pred.double(), target)
    attributemask_acc = compute_accuracy(node_pred, target)
    return attributemask_loss, attributemask_acc
