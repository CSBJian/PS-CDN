import torch
from torch.nn import Module, Linear
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class ConcatSquashLinearSA(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinearSA, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

        self._hyper_k = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_v = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_v.weight = self._hyper_k.weight
        self._hyper_v.bias = self._hyper_k.bias
        self.softmax = nn.Softmax(dim=-1)

        self.norm = nn.BatchNorm1d(dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        x = self._layer(x)
        shape_k = self._hyper_k(ctx).permute(0, 2, 1)
        shape_v = self._hyper_v(ctx)
        energy = torch.bmm(shape_k, shape_v)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        x_r = torch.bmm(x, attention)
        x = x + x_r
        ret = x * gate + bias
        return ret


class ConcatSquashLinearSA2(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinearSA2, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)
        self._hyper_k = Linear(dim_ctx, dim_in, bias=False)
        self._hyper_v = Linear(dim_ctx, dim_in, bias=False)
        self._hyper_v.weight = self._hyper_k.weight
        self._hyper_v.bias = self._hyper_k.bias
        self.softmax = nn.Softmax(dim=-1)

        self.norm = nn.BatchNorm1d(dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        shape_k = self._hyper_k(ctx).permute(0, 2, 1)
        shape_v = self._hyper_v(ctx)
        energy = torch.bmm(shape_k, shape_v)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        x_r = torch.bmm(x, attention)
        x = x + x_r
        x = self._layer(x)
        ret = x * gate + bias
        return ret


class ConcatSquashLinearSA3(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinearSA3, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)
        self._shape_k = Linear(dim_ctx-3, dim_out, bias=False)
        self._shape_v = Linear(dim_ctx-3, dim_out, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.trans_conv = nn.Linear(dim_out, dim_out)
        self.after_norm = nn.BatchNorm1d(2048)
        self.act = nn.ReLU()

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        x = self._layer(x)
        shape = ctx[:, :, 0:ctx.size(2)-3]
        shape_k = self._shape_k(shape).permute(0, 2, 1)
        shape_v = self._shape_v(shape)
        energy = torch.bmm(shape_k, shape_v)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        x_r = torch.bmm(x, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        ret = x * gate + bias
        return ret


class ConcatSquashLinearSA4(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinearSA4, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self.bias = Linear(dim_ctx, dim_out)
        self.gate = Linear(dim_ctx, dim_out)
        self.bias1 = Linear(dim_ctx, dim_out)
        self.gate1 = Linear(dim_ctx, dim_out)
        self.bias2 = Linear(dim_ctx, dim_out)
        self.gate2 = Linear(dim_ctx, dim_out)
        self._shape_k = Linear(dim_ctx-3, dim_out, bias=False)
        self._shape_v = Linear(dim_ctx-3, dim_out, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ctx, x):
        gate = self.gate(ctx)
        bias = self.bias(ctx)
        gate1 = self.gate1(ctx)
        bias1 = self.bias1(ctx)
        gate2 = self.gate2(ctx)
        bias2 = self.bias2(ctx)
        x = self._layer(x)
        shortcut = x
        x = x+x*gate+bias
        shape = ctx[:, :, 0:ctx.size(2)-3]
        shape_k = self._shape_k(shape).permute(0, 2, 1)
        shape_v = self._shape_v(shape)
        energy = torch.bmm(shape_k, shape_v)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        x_r = torch.bmm(x, attention)
        x = x + x_r
        x = shortcut + x *gate1 + bias1
        ret =x + x * gate2 + bias2
        return ret

def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)

def lr_func(epoch):
    if epoch <= start_epoch:
        return 1.0
    elif epoch <= end_epoch:
        total = end_epoch - start_epoch
        delta = epoch - start_epoch
        frac = delta / total
        return (1-frac) * 1.0 + frac * (end_lr / start_lr)
    else:
        return end_lr / start_lr
