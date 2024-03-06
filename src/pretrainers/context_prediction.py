""" GRAPH SSL Pre-Training via Context Prediction (CP)
i.e., maps nodes in similar structural contexts to closer embeddings
Ref Paper: Sec. 3.1.1 and Appendix G of
            https://arxiv.org/abs/1905.12265 ;
Ref Code: ${GitHub_Repo}/chem/pretrain_contextpred.py """


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from config.training_config import TrainingConfig
from logger import CombinedLogger
from models.context_prediction import ContextPredictionModel
from pretrainers.pretrainer import PreTrainer
from util import cycle_idx, get_lr


class ContextPredictionPreTrainer(PreTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        model: ContextPredictionModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: CombinedLogger,
    ):
        super(ContextPredictionPreTrainer, self).__init__(
            config=config,
            model=model,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def validate_model(self, val_data_loader: DataLoader):
        pass

    def train_for_one_epoch(self, train_data_loader: DataLoader):
        self.model.train()
        contextpred_loss_accum, contextpred_acc_accum = 0, 0
        self.logger.train(num_batches=len(train_data_loader))

        for step, batch in enumerate(train_data_loader):
            batch = batch.to(self.device)
            contextpred_loss, contextpred_acc = self.do_ContextPred(batch=batch)
            loss_float = contextpred_loss.detach().cpu().item()
            ssl_loss = contextpred_loss
            self.optimizer.zero_grad()
            ssl_loss.backward()
            self.optimizer.step()
            self.logger(
                ssl_loss.cpu().item(),
                contextpred_acc,
                batch.num_graphs,
                get_lr(self.optimizer),
            )
            contextpred_loss_accum += loss_float
            contextpred_acc_accum += contextpred_acc

        return contextpred_loss_accum / len(train_data_loader)
        # return contextpred_loss_accum / len(train_data_loader), \
        #        contextpred_acc_accum / len(train_data_loader)

    def do_ContextPred(self, batch: Batch):
        # creating substructure representation
        substruct_repr = self.model.forward_substruct_model(
            x=batch.x_substruct,
            edge_index=batch.edge_index_substruct,
            edge_attr=batch.edge_attr_substruct,
        )[batch.center_substruct_idx]

        # substruct_repr = molecule_substruct_model(
        #     batch.x_substruct, batch.edge_index_substruct,
        #     batch.edge_attr_substruct)[batch.center_substruct_idx]

        # create positive context representation
        # readout -> global_mean_pool by default

        context_repr = self.model.forward_context_repr(
            x=batch.x_context,
            edge_index=batch.edge_index_context,
            edge_attr=batch.edge_attr_context,
            overlap_context_substruct_idx=batch.overlap_context_substruct_idx,
            batch_overlapped_context=batch.batch_overlapped_context,
        )

        # negative contexts are obtained by shifting
        # the indices of context embeddings
        neg_context_repr = torch.cat(
            [
                context_repr[cycle_idx(len(context_repr), i + 1)]
                for i in range(self.config.contextpred_neg_samples)
            ],
            dim=0,
        )

        num_neg = self.config.contextpred_neg_samples

        pred_pos = torch.sum(substruct_repr * context_repr, dim=1)
        pred_neg = torch.sum(
            substruct_repr.repeat((num_neg, 1)) * neg_context_repr, dim=1
        )

        loss_pos = self.criterion(
            pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double()
        )
        loss_neg = self.criterion(
            pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double()
        )

        contextpred_loss = loss_pos + num_neg * loss_neg

        num_pred = len(pred_pos) + len(pred_neg)
        contextpred_acc = (
            torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()
        ) / num_pred
        contextpred_acc = contextpred_acc.detach().cpu().item()

        return contextpred_loss, contextpred_acc


if __name__ == "__main__":
    pass
