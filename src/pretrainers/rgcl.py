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

import gc, torch
from config.training_config import TrainingConfig
from pretrainers.pretrainer import PreTrainer
from torch_geometric.loader import DataLoader

# from torch.utils.data import DataLoader
from logger import CombinedLogger
from models.rgcl import RGCLModel
from copy import deepcopy
from util import get_lr


class RGCLPreTrainer(PreTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        model: RGCLModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: CombinedLogger,
    ):
        super(RGCLPreTrainer, self).__init__(
            config=config,
            model=model,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )

    # def train_for_one_epoch(self, train_data_loader: DataLoader) -> float:
    def train_for_one_epoch(self, dataset) -> float:
        dataset.aug = "none"
        loader = DataLoader(dataset, batch_size=2048, num_workers=4, shuffle=False)
        self.model.eval()
        torch.set_grad_enabled(False)

        for step, batch in enumerate(loader):
            node_index_start = step * 2048
            node_index_end = min(node_index_start + 2048 - 1, len(dataset) - 1)
            batch = batch.to(self.device)
            node_imp = self.model.node_imp_estimator(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            ).detach()
            dataset.node_score[
                dataset.slices["x"][node_index_start] : dataset.slices["x"][
                    node_index_end + 1
                ]
            ] = torch.squeeze(node_imp.half())

        dataset1 = deepcopy(dataset)
        dataset1 = dataset1.shuffle()
        dataset2 = deepcopy(dataset1)
        dataset3 = deepcopy(dataset1)

        dataset1.aug, dataset1.aug_ratio = "dropN", 0.2
        dataset2.aug, dataset2.aug_ratio = "dropN", 0.2
        dataset3.aug, dataset3.aug_ratio = "dropN" + "_cp", 0.2

        loader1 = DataLoader(
            dataset1,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        loader2 = DataLoader(
            dataset2,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        loader3 = DataLoader(
            dataset3,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        train_loss_accum = 0
        ra_loss_accum = 0
        cp_loss_accum = 0

        torch.set_grad_enabled(True)
        self.model.train()
        self.logger.train(num_batches=len(loader1))

        for step, batch in enumerate(zip(loader1, loader2, loader3)):
            batch1, batch2, batch3 = batch
            batch1 = batch1.to(self.device)
            batch2 = batch2.to(self.device)
            batch3 = batch3.to(self.device)

            self.optimizer.zero_grad()

            x1 = self.model.forward_cl(
                batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch
            )
            x2 = self.model.forward_cl(
                batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch
            )
            x3 = self.model.forward_cl(
                batch3.x, batch3.edge_index, batch3.edge_attr, batch3.batch
            )

            ra_loss, cp_loss, loss = self.model.loss_ra(x1, x2, x3)

            loss.backward()
            self.optimizer.step()
            loss_float = float(loss.detach().cpu().item())
            train_loss_accum += loss_float
            ra_loss_accum += float(ra_loss.detach().cpu().item())
            cp_loss_accum += float(cp_loss.detach().cpu().item())
            self.logger(loss_float, 0.0, batch1.num_graphs, get_lr(self.optimizer))
            # del dataset1, dataset2, dataset3
            # gc.collect()

        # return train_loss_accum/(step+1), ra_loss_accum/(step+1), cp_loss_accum/(step+1)
        return train_loss_accum / (step + 1)

    def validate_model(self, val_data_loader) -> float:
        # self.logger.eval(num_batches=len(val_data_loader))
        self.logger.eval(num_batches=1)
        self.logger(0.0, 0.0, 1)
        return 0.0
