import numpy as np
import torch
from torch.utils.data import DataLoader

from config.training_config import TrainingConfig
from logger import CombinedLogger
from models.joao_v2 import JOAOv2Model
from pretrainers.pretrainer import PreTrainer
from util import get_lr


class JOAOv2PreTrainer(PreTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        model: JOAOv2Model,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: CombinedLogger,
    ):
        super(JOAOv2PreTrainer, self).__init__(
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

        aug_prob = train_data_loader.dataset.aug_prob
        n_aug = np.random.choice(25, 1, p=aug_prob)[0]
        n_aug1, n_aug2 = n_aug // 5, n_aug % 5

        for step, (_, batch1, batch2) in enumerate(train_data_loader):
            batch1 = batch1.to(self.device)
            batch2 = batch2.to(self.device)

            x1 = self.model.forward_cl(
                batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch, n_aug1
            )
            x2 = self.model.forward_cl(
                batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, n_aug2
            )
            loss = self.model.loss_cl(x1, x2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_float = float(loss.detach().cpu().item())
            train_loss_accum += loss_float
            self.logger(loss_float, 0.0, batch1.num_graphs, get_lr(self.optimizer))

        # joaov2
        aug_prob = train_data_loader.dataset.aug_prob
        # TODO: Can we replace the below constants (25, 10, etc.) with arguments in the config?
        loss_aug = np.zeros(25)
        for n in range(25):
            _aug_prob = np.zeros(25)
            _aug_prob[n] = 1
            train_data_loader.dataset.set_augProb(_aug_prob)

            count, count_stop = (
                0,
                len(train_data_loader.dataset) // (train_data_loader.batch_size * 10)
                + 1,
            )
            # for efficiency, we only use around 10% of data to estimate the loss
            n_aug1, n_aug2 = n // 5, n % 5
            with torch.no_grad():
                for step, (_, batch1, batch2) in enumerate(train_data_loader):
                    batch1 = batch1.to(self.device)
                    batch2 = batch2.to(self.device)

                    x1 = self.model.forward_cl(
                        batch1.x,
                        batch1.edge_index,
                        batch1.edge_attr,
                        batch1.batch,
                        n_aug1,
                    )
                    x2 = self.model.forward_cl(
                        batch2.x,
                        batch2.edge_index,
                        batch2.edge_attr,
                        batch2.batch,
                        n_aug2,
                    )
                    loss = self.model.loss_cl(x1, x2)
                    loss_aug[n] += loss.item()
                    count += 1
                    if count == count_stop:
                        break
            loss_aug[n] /= count

        # view selection, projected gradient descent,
        # reference: https://arxiv.org/abs/1906.03563
        beta = 1
        gamma = self.config.gamma_joaov2

        b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1 / 25))
        mu_min, mu_max = b.min() - 1 / 25, b.max() - 1 / 25
        mu = (mu_min + mu_max) / 2

        # bisection method
        while abs(np.maximum(b - mu, 0).sum() - 1) > 1e-2:
            if np.maximum(b - mu, 0).sum() > 1:
                mu_min = mu
            else:
                mu_max = mu
            mu = (mu_min + mu_max) / 2

        aug_prob = np.maximum(b - mu, 0)
        aug_prob /= aug_prob.sum()
        train_data_loader.dataset.set_augProb(aug_prob=aug_prob)
        self.logger.log_value_dict({"aug_prob": aug_prob})
        return train_loss_accum / (step + 1)

    # TODO
    def validate_model(self, val_data_loader: DataLoader) -> float:
        # self.logger.eval(num_batches=len(val_data_loader))
        self.logger.eval(num_batches=1)
        self.logger(0.0, 0.0, 1)
        return 0.0
