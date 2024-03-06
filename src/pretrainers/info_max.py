""" GRAPH SSL Pre-Training via InfoGraph [InfoGraph]
i.e., maps nodes in similar structural contexts to closer embeddings
Ref Paper: Sec. 5.2 and Appendix G of
            https://arxiv.org/abs/1905.12265 ;
           which is adapted from
            https://arxiv.org/abs/1809.10341 ;
Ref Code: ${GitHub_Repo}/chem/pretrain_deepgraphinfomax.py """

import torch
import torch.nn as nn
from torch_geometric.data import Batch, DataLoader
from torch_geometric.nn import global_mean_pool

from config.training_config import TrainingConfig
from logger import CombinedLogger
from models.info_max import InfoMaxModel
from pretrainers.pretrainer import PreTrainer
from util import cycle_idx, get_lr


class InfoMaxPreTrainer(PreTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        model: InfoMaxModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: CombinedLogger,
    ):
        super(InfoMaxPreTrainer, self).__init__(
            config=config,
            model=model,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )
        # TODO: perhaps we can add the below attributes as arguments?
        self.molecule_readout_func = global_mean_pool
        self.criterion = nn.BCEWithLogitsLoss()

    def train_for_one_epoch(self, train_data_loader: DataLoader) -> float:
        self.model.train()
        infograph_loss_accum, infograph_acc_accum = 0, 0
        self.logger.train(num_batches=len(train_data_loader))

        for step, batch in enumerate(train_data_loader):
            batch = batch.to(self.device)
            node_emb, pooled_emb = self.model.forward_embedding(batch)
            infograph_loss, infograph_acc = do_InfoGraph(
                node_repr=node_emb,
                batch=batch,
                molecule_repr=pooled_emb,
                criterion=self.criterion,
                model=self.model,
            )

            self.optimizer.zero_grad()
            infograph_loss.backward()
            self.optimizer.step()
            infograph_loss = infograph_loss.detach().cpu().item()
            infograph_loss_accum += infograph_loss
            infograph_acc_accum += infograph_acc
            self.logger(
                infograph_loss, infograph_acc, batch.num_graphs, get_lr(self.optimizer)
            )

        return infograph_loss_accum / (step + 1)

    # TODO
    def validate_model(self, val_data_loader: DataLoader) -> float:
        # self.logger.eval(num_batches=len(val_data_loader))
        self.logger.eval(num_batches=1)
        self.logger(0.0, 0.0, 1)
        return 0.0


def do_InfoGraph(
    node_repr: torch.Tensor,
    molecule_repr: torch.Tensor,
    batch: Batch,
    criterion: torch.nn.Module,
    model: InfoMaxModel,
):
    summary_repr = torch.sigmoid(molecule_repr)
    positive_expanded_summary_repr = summary_repr[batch.batch]
    shifted_summary_repr = summary_repr[cycle_idx(len(summary_repr), 1)]
    negative_expanded_summary_repr = shifted_summary_repr[batch.batch]

    positive_score = model.infograph_discriminator_SSL_model(
        node_repr, positive_expanded_summary_repr
    )
    negative_score = model.infograph_discriminator_SSL_model(
        node_repr, negative_expanded_summary_repr
    )
    infograph_loss = criterion(
        positive_score, torch.ones_like(positive_score)
    ) + criterion(negative_score, torch.zeros_like(negative_score))

    num_sample = float(2 * len(positive_score))
    infograph_acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(
        torch.float32
    ) / num_sample
    infograph_acc = infograph_acc.detach().cpu().item()

    return infograph_loss, infograph_acc
