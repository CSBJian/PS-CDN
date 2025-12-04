import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import Module, Parameter, ModuleList
from .common import *


class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(3, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 3, context_dim+3)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class PointwiseNetSA4(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinearSA4(3, 128, context_dim+3),
            ConcatSquashLinearSA4(128, 256, context_dim+3),
            ConcatSquashLinearSA4(256, 512, context_dim+3),
            ConcatSquashLinearSA4(512, 256, context_dim+3),
            ConcatSquashLinearSA4(256, 128, context_dim+3),
            ConcatSquashLinearSA4(128, 3, context_dim+3)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)

        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)
        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]



#    新方法

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ShapeAttention(Module):
    def __init__(self, dim_in, dim_out, dim_time, dim_shape):
        super(ShapeAttention, self).__init__()
        # 正式代码
        self.x_embedding = nn.Conv1d(dim_in, dim_out,1)
        self.time_embedding = nn.Conv1d(dim_time, dim_out,1)
        self.shape_embedding = nn.Conv1d(dim_shape, dim_out,1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(dim_out, 3 * dim_out, 1)
        )
        self.shape_kv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(dim_out, 2 * dim_out, 1)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.time_transfer = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(dim_out, 3 * dim_out, 1)
        )

        self.MLP = nn.Sequential(
            nn.Conv1d(dim_out, dim_out * 2,1),
            nn.SiLU(),
            nn.Conv1d(dim_out * 2, dim_out,1)
        )
        self.norm1 = nn.BatchNorm1d(dim_out)
        self.norm2 = nn.BatchNorm1d(dim_out)

    def forward(self, time, x, shape):

        x = self.x_embedding(x)
        shortcut = x
        # shape  和  time  确定去除的形状信息 更好的建模噪声分布
        # 1. 将shape和time编码融合 获取当前的shape信息比重
        #     shape = shape * time
        time = self.time_embedding(time.permute(0, 2, 1))
        shape = self.shape_embedding(shape.permute(0, 2, 1))
        time_hype_shape = time * shape
        gate_global, shift_global, scale_global = self.adaLN_modulation(time_hype_shape).chunk(3, dim=1)
        #     x = x + x * scale(shape) + shift(shape)   全局信息变换
        x1 = modulate(self.norm1(x), shift_global, scale_global)

        # 2. 使用注意力机制  将shape信息  从 x中去除  得到纯粹的噪声信息     形状注意力机制
        #     x = x - x * atten
        # 得到去除噪声的信息
        shape_k, shape_v = self.shape_kv(shape).chunk(2, dim=1)
        shape_v = shape_v.permute(0, 2, 1)
        energy = torch.bmm(shape_k, shape_v)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        x_shape = torch.bmm(attention, x1)
        x1 = x1 - x_shape
        x = x + gate_global * x1

        # 3. 使用time信息变换 得到真实的噪声估计
        #     x = x + x * scale(time) + shift(time)     局部信息变换
        gate_local, shift_local, scale_local = self.time_transfer(time).chunk(3, dim=1)
        ret = shortcut + gate_local * self.MLP(modulate(self.norm2(x), shift_local, scale_local))
        return ret


class PointwiseNetAtten(Module):

    def __init__(self, point_dim, context_dim, residual=1):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ShapeAttention(3, 128, 3, context_dim),
            ShapeAttention(128, 256, 3, context_dim),
            ShapeAttention(256, 512, 3, context_dim),
            ShapeAttention(512, 256, 3, context_dim),
            ShapeAttention(256, 128, 3, context_dim),
            ShapeAttention(128, 3, 3, context_dim)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        x= x.permute(0, 2, 1)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        # ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(time_emb, out, context)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return (x + out).permute(0, 2, 1)
        else:
            return out.permute(0, 2, 1)


