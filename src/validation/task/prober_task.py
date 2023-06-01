from typing import Dict, Union
import wandb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.validation_config import ValidationConfig

# from logger import CombinedLogger
from util import get_lr
from datasets import MoleculeDataset
from validation.dataset import ProberDataset
from validation.task import TrainValTestTask


class ProberTask(TrainValTestTask):
    def __init__(
        self,
        config: ValidationConfig,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        # logger: CombinedLogger,
        train_dataset: Union[MoleculeDataset, ProberDataset],
        val_dataset: Union[MoleculeDataset, ProberDataset],
        test_dataset: Union[MoleculeDataset, ProberDataset],
        criterion_type: str = "mse",
    ):
        super(ProberTask, self).__init__(
            config=config,
            model=model,
            device=device,
            optimizer=optimizer,
            # logger=logger,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            criterion_type=config.criterion_type,
        )
        if config.val_task == "finetune":
            from torch_geometric.loader import DataLoader
        else:
            from torch.utils.data import DataLoader
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=config.batch_size, shuffle=False
        )

        if criterion_type == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
        elif criterion_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        elif criterion_type == "ce":
            self.criterion = nn.CrossEntropyLoss(reduction="mean")
        else:
            raise Exception("Unknown criterion {}".format(criterion_type))

    def train(self) -> float:
        pass

    def run(self):
        for _ in tqdm(range(self.config.epochs)):
            if self.config.val_task == "finetune":
                train_loss = self.train_finetune_step()
            else:
                train_loss = self.train_step()

            if self.config.probe_task == "downstream":
                # train_score = self._eval_roc(loader=self.train_loader)
                val_score = self._eval_roc(loader=self.val_loader)
                test_score = self._eval_roc(loader=self.test_loader)
            else:
                val_score = self.eval_val_dataset()
                test_score = self.eval_test_dataset()

            results_dict = {
                "train_loss": train_loss,
                "val_score": val_score,
                "test_score": test_score,
            }
            wandb.log(results_dict)

        print(results_dict)
        print("\n\n\n")
        return

    def train_step(self) -> float:
        self.model.train()
        total_loss = 0

        for _, batch in enumerate(self.train_loader):

            if self.config.probe_task == "downstream":
                pred = self.model(batch["representation"].to(self.device))
                y = batch["label"].to(torch.float32).to(self.device)

                # Whether y is non-null or not.
                is_valid = y**2 > 0
                # Loss matrix
                loss_mat = self.criterion(pred.double(), (y + 1) / 2)
                # loss matrix after removing null target
                loss_mat = torch.where(
                    is_valid,
                    loss_mat,
                    torch.zeros(loss_mat.shape).to(self.device).to(loss_mat.dtype),
                )

                self.optimizer.zero_grad()
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach().item()

            else:
                pred = self.model(batch["representation"].to(self.device)).squeeze()
                y = batch["label"].to(torch.float32).to(self.device)
                loss = self.criterion(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach().item()

        return total_loss / len(self.train_loader)

    def train_finetune_step(self) -> float:

        self.model.train()
        total_loss = 0

        if self.config.probe_task != "downstream":
            raise NotImplementedError

        for _, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.view(pred.shape).to(torch.float64)

            is_valid = y**2 > 0
            loss_mat = self.criterion(pred.double(), (y + 1) / 2)
            loss_mat = torch.where(
                is_valid,
                loss_mat,
                torch.zeros(loss_mat.shape).to(self.device).to(loss_mat.dtype),
            )

            self.optimizer.zero_grad()
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach().item()

        return total_loss / len(self.train_loader)

    def _eval(self, loader: DataLoader) -> float:
        self.model.eval()
        y_true, y_pred = [], []
        for _, batch in enumerate(loader):
            inputs = batch["representation"].to(self.device)
            with torch.no_grad():
                pred = self.model(inputs).squeeze()
            y_true.append(batch["label"])
            y_pred.append(pred.detach().cpu())
        y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
        if self.criterion is nn.BCELoss:
            y_pred = nn.Sigmoid(y_pred)
        return self.criterion(y_pred, y_true).item()

    def _eval_roc(self, loader: DataLoader) -> float:

        self.model.eval()
        y_true, y_pred = [], []

        for _, batch in enumerate(loader):

            with torch.no_grad():
                if self.config.val_task == "prober":
                    inputs = batch["representation"].to(self.device)
                    pred = self.model(inputs)
                    true = batch["label"].to(torch.float32).to(self.device)
                elif self.config.val_task == "finetune":
                    batch = batch.to(self.device)
                    pred = self.model(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch
                    )
                    true = batch.y.view(pred.shape)

            y_true.append(true)
            y_pred.append(pred)
            # y_pred.append(pred.view(true.shape))

        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            # if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            #     roc_list.append(roc_auc_score(y_true[:, i], y_pred[:, i]))

            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                is_valid = y_true[:, i] ** 2 > 0
                roc_list.append(
                    roc_auc_score((y_true[is_valid, i] + 1) / 2, y_pred[is_valid, i])
                )

        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

        return sum(roc_list) / len(roc_list)

    def eval_train_dataset(self) -> float:
        return self._eval(loader=self.train_loader)

    def eval_val_dataset(self) -> float:
        return self._eval(loader=self.val_loader)

    def eval_test_dataset(self) -> float:
        return self._eval(loader=self.test_loader)
