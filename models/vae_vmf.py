import torch
from torch.nn import Module
# from hyperspherical_vae import VonMisesFisher
# from hyperspherical_vae import HypersphericalUniform

# from .common import *
from .encoders import *
from .diffusion import *
from .VMF import *


class VMFVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.VMF = VMFPointNetEncoder(args)

        if args.backbone == "pw":
            self.diffusion = DiffusionPoint(
                net=PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
                var_sched=VarianceSchedule(
                    num_steps=args.num_steps,
                    beta_1=args.beta_1,
                    beta_T=args.beta_T,
                    mode=args.sched_mode
                )
            )
        if args.backbone == "sa":
            self.diffusion = DiffusionPoint(
                net=PointwiseNetSA4(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
                var_sched=VarianceSchedule(
                    num_steps=args.num_steps,
                    beta_1=args.beta_1,
                    beta_T=args.beta_T,
                    mode=args.sched_mode
                )
            )


    def get_loss(self, x, writer=None, it=None, kl_weight=1.0):
        """
        Args:
            angle:  Input point clouds, (B, N, 2).
            r:  Input point clouds, (B, N, 1).
        """
        batch_size, _, _ = x.size()
        device = x.device

        z_mu, z_k = self.VMF(x)
        q_z = VonMisesFisher(z_mu, z_k)
        z_vmf = q_z.rsample()
        p_z = HypersphericalUniform(self.args.latent_dim - 1, device=device)
        entropy = q_z.entropy()
        loss_prior = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

        loss_recons = self.diffusion.get_loss(x, z_vmf)
        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), it)
            writer.add_scalar('train/loss_prior', -loss_prior, it)
            writer.add_scalar('train/loss_recons', loss_recons, it)

        return loss

    def sample(self, z, num_points, flexibility, truncate_std=None):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples

