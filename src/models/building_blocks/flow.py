import torch
import torch.distributions as distrib
import torch.distributions.transforms as transform
import torch.nn as nn
import torch.nn.functional as F


class Flow(transform.Transform, nn.Module):
    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)

    def __hash__(self):
        return nn.Module.__hash__(self)


class PlanarFlow(Flow):
    def __init__(self, dim, h=torch.tanh, hp=(lambda x: 1 - torch.tanh(x) ** 2)):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.h = h
        self.hp = hp
        self.init_parameters()

    def _call(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.h(f_z)

    def log_abs_det_jacobian(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        psi = self.hp(f_z) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-9)


class RadialFlow(Flow):
    def __init__(self, dim):
        super(RadialFlow, self).__init__()
        self.z0 = nn.Parameter(torch.Tensor(1, dim))
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.dim = dim
        self.init_parameters()

    def _call(self, z):
        r = torch.norm(z - self.z0, dim=1).unsqueeze(1)
        h = 1 / (self.alpha + r)
        return z + (self.beta * h * (z - self.z0))

    def log_abs_det_jacobian(self, z):
        r = torch.norm(z - self.z0, dim=1).unsqueeze(1)
        h = 1 / (self.alpha + r)
        hp = -1 / (self.alpha + r) ** 2
        bh = self.beta * h
        det_grad = ((1 + bh) ** self.dim - 1) * (1 + bh + self.beta * hp * r)
        return torch.log(det_grad.abs() + 1e-9)


class PReLUFlow(Flow):
    def __init__(self, dim):
        super(PReLUFlow, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([1]))
        self.bijective = True

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(0.01, 0.99)

    def _call(self, z):
        return torch.where(z >= 0, z, torch.abs(self.alpha) * z)

    def _inverse(self, z):
        return torch.where(z >= 0, z, torch.abs(1.0 / self.alpha) * z)

    def log_abs_det_jacobian(self, z):
        I = torch.ones_like(z)
        J = torch.where(z >= 0, I, self.alpha * I)
        log_abs_det = torch.log(torch.abs(J) + 1e-5)
        return torch.sum(log_abs_det, dim=1)


class BatchNormFlow(Flow):
    def __init__(self, dim, momentum=0.95, eps=1e-5):
        super(BatchNormFlow, self).__init__()
        # Running batch statistics
        self.r_mean = torch.zeros(dim)
        self.r_var = torch.ones(dim)
        # Momentum
        self.momentum = momentum
        self.eps = eps
        # Trainable scale and shift (cf. original paper)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def _call(self, z):
        if self.training:
            # Current batch stats
            self.b_mean = z.mean(0)
            self.b_var = (z - self.b_mean).pow(2).mean(0) + self.eps
            # Running mean and var
            self.r_mean = (
                self.momentum * self.r_mean + (1 - self.momentum) * self.b_mean
            )
            self.r_var = self.momentum * self.r_var + (1 - self.momentum) * self.b_var
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var
        x_hat = (z - mean) / var.sqrt()
        y = self.gamma * x_hat + self.beta
        return y

    def _inverse(self, x):
        if self.training:
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var
        x_hat = (z - self.beta) / self.gamma
        y = x_hat * var.sqrt() + mean
        return y

    def log_abs_det_jacobian(self, z):
        # Here we only need the variance
        mean = z.mean(0)
        var = (z - mean).pow(2).mean(0) + self.eps
        log_det = torch.log(self.gamma) - 0.5 * torch.log(var + self.eps)
        return torch.sum(log_det, -1)


class AffineFlow(Flow):
    def __init__(self, dim):
        super(AffineFlow, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(dim, dim))
        nn.init.orthogonal_(self.weights)

    def _call(self, z):
        return z @ self.weights

    def _inverse(self, z):
        return z @ torch.inverse(self.weights)

    def log_abs_det_jacobian(self, z):
        return torch.slogdet(self.weights)[-1].unsqueeze(0).repeat(z.size(0), 1)


class NormalizingFlow(nn.Module):
    def __init__(self, dim, blocks, flow_length, density):
        super(NormalizingFlow, self).__init__()
        biject = []
        for f in range(flow_length):
            for b_flow in blocks:
                biject.append(b_flow(dim))
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.base_density = density
        self.final_density = distrib.TransformedDistribution(density, self.transforms)
        self.log_det = []

    def forward(self, z):
        self.log_det = []
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)
        return z, self.log_det


if __name__ == "__main__":
    flow_model = "mlp"
    if flow_model == "planar":
        blocks = [PlanarFlow]
    elif flow_model == "radial":
        blocks = [RadialFlow]
    elif flow_model == "affine":
        blocks = [AffineFlow]
    elif flow_model == "mlp":
        blocks = [AffineFlow, BatchNormFlow, PReLUFlow]
    else:
        blocks = None

    flow = NormalizingFlow(
        dim=2,
        blocks=blocks,
        flow_length=8,
        density=distrib.MultivariateNormal(torch.zeros(2), torch.eye(2)),
    )

    import numpy as np
    import torch.optim as optim

    def density_ring(z):
        z1, z2 = torch.chunk(z, chunks=2, dim=1)
        norm = torch.sqrt(z1**2 + z2**2)
        exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
        exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
        u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)
        return torch.exp(-u)

    def loss(density, zk, log_jacobians):
        sum_of_log_jacobians = sum(log_jacobians)
        return (-sum_of_log_jacobians - torch.log(density(zk) + 1e-9)).mean()

    optimizer = optim.Adam(flow.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)

    x = np.linspace(-5, 5, 1000)
    z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
    z = np.reshape(z, [z.shape[0] * z.shape[1], -1])

    ref_distrib = distrib.MultivariateNormal(torch.zeros(2), torch.eye(2))
    for it in range(10001):
        # Draw a sample batch from Normal
        samples = ref_distrib.sample((512,))
        # Evaluate flow of transforms
        zk, log_jacobians = flow(samples)
        # Evaluate loss and backprop
        optimizer.zero_grad()
        loss_v = loss(density_ring, zk, log_jacobians)
        loss_v.backward()
        optimizer.step()
        scheduler.step()
        if it % 1000 == 0:
            print("Loss (it. %i) : %f" % (it, loss_v.item()))
            # Draw random samples
            samples = ref_distrib.sample((int(1e5),))
            # Evaluate flow and plot
            zk, _ = flow(samples)
