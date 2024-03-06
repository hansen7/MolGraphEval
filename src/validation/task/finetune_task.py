# import pdb
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.validation_config import ValidationConfig

# from logger import CombinedLogger
# from util import get_lr
from validation.dataset import ProberDataset
from validation.task import TrainValTestTask


class ProberTask(TrainValTestTask):
    def __init__(
        self,
        config: ValidationConfig,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        train_dataset: ProberDataset,
        val_dataset: ProberDataset,
        test_dataset: ProberDataset,
        criterion_type: str = "mse",
    ):
        super(ProberTask, self).__init__(
            config=config,
            model=model,
            device=device,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            criterion_type=config.criterion_type,
        )
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

        self.train_loss = []
        self.train_score, self.eval_score, self.test_score = [], [], []

    def run(self, model: nn.Module = None, device: torch.device = None) -> Dict:
        # final_train_loss = self.eval_train_dataset()
        final_train_loss = self.train()
        final_val_loss = self.eval_val_dataset()
        final_test_loss = self.eval_test_dataset()
        results_dict = {
            "train_loss": final_train_loss,
            "val_loss": final_val_loss,
            "test_loss": final_test_loss,
        }
        print(results_dict)
        return results_dict

    def train(self) -> float:
        for _ in tqdm(range(self.config.epochs)):
            train_loss = self.train_step()
            # val_loss = self.eval_val_dataset()
        return train_loss

    def train_step(self) -> float:
        self.model.train()
        total_loss = 0
        # self.logger.train(num_batches=len(self.train_loader))

        for _, batch in enumerate(self.train_loader):
            # pdb.set_trace()
            pred = self.model(batch["representation"].to(self.device)).squeeze()
            y = batch["label"].to(torch.float32).to(self.device)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # loss_float = loss.detach().item()
            total_loss += loss.detach().item()

        return total_loss / len(self.train_loader)

    def _eval(self, loader: DataLoader) -> float:
        self.model.eval()
        total_val_loss = 0.0
        for _, batch in enumerate(loader):
            with torch.no_grad():
                inputs = batch["representation"].to(self.device)
                pred = self.model(inputs).squeeze()
                true_target = batch["label"].to(torch.float32).to(self.device)
                val_loss = self.criterion(pred, true_target).item()
                total_val_loss += val_loss
        return total_val_loss

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
