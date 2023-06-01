import sys

import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F

from .flow import (
    AffineFlow,
    BatchNormFlow,
    NormalizingFlow,
    PlanarFlow,
    PReLUFlow,
    RadialFlow,
)

sys.path.insert(0, "../")
from util import do_CL


def L1_loss(p, z, average=True):
    loss = torch.abs(p - z)
    loss = loss.sum(dim=1)
    if average:
        loss = loss.mean()
    return loss


def L2_loss(p, z, average=True):
    loss = (p - z) ** 2
    loss = loss.sum(dim=1)
    if average:
        loss = loss.mean()
    return loss


def cosine_similarity(p, z, average=True):
    p = F.normalize(p, p=2, dim=1)
    z = F.normalize(z, p=2, dim=1)
    loss = -(p * z).sum(dim=1)
    if average:
        loss = loss.mean()
    return loss


class AutoEncoder(nn.Module):
    def __init__(self, emb_dim, loss, detach_target):
        super(AutoEncoder, self).__init__()
        self.loss = loss
        self.emb_dim = emb_dim
        self.detach_target = detach_target

        if loss == "l1":
            self.criterion = nn.L1Loss()
        elif loss == "l2":
            self.criterion = nn.MSELoss()
        elif loss == "cosine":
            self.criterion = cosine_similarity
        else:
            raise ValueError("criterion must be l1, l2 or cosine")
        self.fc_layers = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        # why not have BN as last FC layer:
        # https://stats.stackexchange.com/questions/361700/
        return

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()
        x = self.fc_layers(x)
        loss = self.criterion(x, y)
        return loss


class VariationalAutoEncoder(nn.Module):
    def __init__(self, emb_dim, loss, detach_target, beta=1):
        super(VariationalAutoEncoder, self).__init__()
        self.detach_target = detach_target
        self.emb_dim = emb_dim
        self.loss = loss
        self.beta = beta
        if loss == "l1":
            self.criterion = nn.L1Loss()
        elif loss == "l2":
            self.criterion = nn.MSELoss()
        elif loss == "cosine":
            self.criterion = cosine_similarity
        else:
            raise ValueError("criterion must be l1, l2 or cosine")
        self.fc_mu = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        return

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decoder(z)

        reconstruction_loss = self.criterion(y_hat, y)
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        # negative reconstruction + KL
        loss = reconstruction_loss + self.beta * kl_loss
        return loss


class NormalizingFlowVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        loss,
        detach_target,
        flow_model,
        flow_length,
        beta=1,
        kl_div_exact=False,
    ):
        super(NormalizingFlowVariationalAutoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.loss = loss
        self.detach_target = detach_target
        self.beta = beta
        self.kl_div_exact = kl_div_exact

        if flow_model == "planar":
            blocks = [PlanarFlow]
        elif flow_model == "radial":
            blocks = [RadialFlow]
        elif flow_model == "affine":
            blocks = [AffineFlow]
        elif flow_model == "mlp":
            blocks = [AffineFlow, BatchNormFlow, PReLUFlow]
        else:
            raise ValueError

        self.criterion = None
        if loss == "l1":
            self.criterion = nn.L1Loss()
        elif loss == "l2":
            self.criterion = nn.MSELoss()
        elif loss == "cosine":
            self.criterion = cosine_similarity

        self.fc_mu = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.emb_dim)

        self.flow = NormalizingFlow(
            dim=emb_dim,
            blocks=blocks,
            flow_length=flow_length,
            density=distrib.MultivariateNormal(
                torch.zeros(emb_dim), torch.eye(emb_dim)
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        return

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()

        B = x.size()[0]
        mu, log_var = self.encode(x)
        z0 = self.reparameterize(mu, log_var)
        zK, list_log_determinant = self.flow(z0)
        y_hat = self.decoder(zK)

        # Here kl_div_exact is preferable,
        # since the sum_log_determinant is not exact expectation.
        if not self.kl_div_exact:
            # log q(z_0)
            log_q_z0 = -0.5 * (
                log_var + (z0 - mu) * (z0 - mu) * log_var.exp().reciprocal()
            )
            # log p(z_k)
            log_p_zk = -0.5 * zK * zK
            # sum of log determinant
            sum_log_determinant = torch.sum(
                torch.stack(
                    [
                        torch.sum(log_determinants)
                        for log_determinants in list_log_determinant
                    ]
                )
            )

            # log q(z_K) - log p(z_K) = log q(z_0) - sum[log det] - log p(z_k)
            log_probability = (
                torch.sum((log_q_z0 - log_p_zk)) - sum_log_determinant
            ) / B
        else:
            # kl_div = log q(z_0) - log p(z_K)
            kl_div = torch.mean(
                -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
            )
            # sum of log determinant
            sum_log_determinant = torch.sum(
                torch.stack(
                    [
                        torch.sum(log_determinants)
                        for log_determinants in list_log_determinant
                    ]
                )
            )

            # log q(z_K) - log p(z_K) = log q(z_0) - sum[log det] - log p(z_k)
            log_probability = kl_div - sum_log_determinant / B

        reconstruction_loss = self.criterion(y_hat, y)

        # negative reconstruction + KL
        loss = reconstruction_loss + self.beta * log_probability

        return loss


class ImportanceWeightedAutoEncoder(VariationalAutoEncoder):
    def __init__(self, emb_dim, loss, detach_target, num_samples, beta=1):
        super(ImportanceWeightedAutoEncoder, self).__init__(
            emb_dim, loss, detach_target, beta
        )
        self.num_samples = num_samples

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        return

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()
        y = y.unsqueeze(1).expand(-1, self.num_samples, -1)

        mu, log_var = self.encode(x)
        mu = mu.unsqueeze(1).expand(-1, self.num_samples, -1)
        log_var = log_var.unsqueeze(1).expand(-1, self.num_samples, -1)

        z, eps = self.reparameterize(mu, log_var)

        B = z.size()[0]
        y_hat = self.decoder(z.view(B * self.num_samples, -1))
        y_hat = y_hat.view(B, self.num_samples, -1)

        reconstruction_loss = ((y_hat - y) ** 2).mean(-1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=2)
        # get importance weights
        log_weight = reconstruction_loss + self.beta * kl_loss
        # rescale the weights (along the sample dim)
        # to be in [0, 1] and sum to 1
        weight = F.softmax(log_weight, dim=-1)
        weight = weight.detach().data

        loss = torch.mean(torch.sum(weight * log_weight, dim=-1), dim=0)
        return loss

    def forward_mark(self, x, y):
        if self.detach_target:
            y = y.detach()
        y = y.unsqueeze(1).expand(-1, self.num_samples, -1)

        mu, log_var = self.encode(x)  # B, dim
        mu = mu.unsqueeze(1).expand(-1, self.num_samples, -1)
        log_var = log_var.unsqueeze(1).expand(-1, self.num_samples, -1)
        z, eps = self.reparameterize(mu, log_var)

        B = z.size()[0]
        y_hat = self.decoder(z.view(B * self.num_samples, -1))
        y_hat = y_hat.view(B, self.num_samples, -1)

        reconstruction_loss = ((y_hat - y) ** 2).mean(-1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=2)
        # get importance weights
        log_weight = reconstruction_loss + self.beta * kl_loss
        # rescale the weights (along the sample dim)
        # to lie in [0, 1] and sum to 1
        weight = F.softmax(log_weight, dim=-1)
        weight = weight.detach().data
        print("weight\n", weight[:3])

        loss = torch.mean(torch.sum(weight * log_weight, dim=-1), dim=0)
        return loss


class EnergyVariationalAutoEncoder(nn.Module):
    def __init__(self, emb_dim, loss, detach_target, args, beta=1):
        super(EnergyVariationalAutoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.loss = loss
        self.detach_target = detach_target
        self.beta = beta

        self.args = args

        self.fc_mu = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.emb_dim)

        return

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        reconstruction_loss, reconstruction_acc = do_CL(z, y, self.args)

        kl_loss = torch.mean(
            -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        # negative reconstruction + KL
        loss = reconstruction_loss + self.beta * kl_loss

        return loss, reconstruction_acc
