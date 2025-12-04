import os
import time
import math
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.vae_vmf import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *


def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./logs_gen/GEN_carvmfpw6_2023_04_27__17_08_29/ckpt_0.000000_1000000.pt')
# parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save_dir', type=str, default='./shapenet15kv2')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='../data/shapenet15kv2.hdf5')
parser.add_argument('--batch_size', type=int, default=64)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_unit', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=3366)
args = parser.parse_args()

# Checkpoint
ckpt = torch.load(args.ckpt)

# Logging
iter = ckpt['others']['optimizer']['state'][0]['step']
# save_dir = os.path.join(args.save_dir, 'GEN_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
save_dir = os.path.join(args.save_dir, 'GEN_Ours_' + ckpt['args'].categories[0] + str(iter) + ckpt['args'].model + ckpt['args'].backbone + str(ckpt['args'].latent_dim))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

seed_all(args.seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=ckpt['args'].categories,
    split='val',
    scale_mode=args.normalize,
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
logger.info('Loading model...')
if ckpt['args'].model == 'gaussian':
    model = GaussianVAE(ckpt['args']).to(args.device)
elif ckpt['args'].model == 'flow':
    model = FlowVAE(ckpt['args']).to(args.device)
elif ckpt['args'].model == 'vmf':
    model = VMFVAE(ckpt['args']).to(args.device)
logger.info(repr(model))
# if ckpt['args'].spectral_norm:
#     add_spectral_norm(model, logger=logger)
model.load_state_dict(ckpt['state_dict'])

# Reference Point Clouds
ref_pcs = []
for i, data in enumerate(test_dset):
    ref_pcs.append(data['pointcloud'].unsqueeze(0))
ref_pcs = torch.cat(ref_pcs, dim=0)
#ref_pcs = 0.5 * torch.stack(ref_pcs, dim=0)

# Generate Point Clouds
gen_pcs = []
for i in tqdm(range(0, math.ceil(len(test_dset) / args.batch_size)), 'Generate'):
    with torch.no_grad():

        if ckpt['args'].model == 'gaussian':
            z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        elif ckpt['args'].model == 'flow':
            z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        elif ckpt['args'].model == 'vmf':
            z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
            z = z / z.norm(dim=-1, keepdim=True)

            #p_z = HypersphericalUniform(ckpt['args'].latent_dim - 1)
            #z = p_z.sample(shape=torch.tensor([args.batch_size, ])).squeeze(1).to(args.device)
        # z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        x = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility)
        gen_pcs.append(x.detach().cpu())
gen_pcs = torch.cat(gen_pcs, dim=0)[:len(test_dset)]
if args.normalize is not None:
    gen_pcs = normalize_point_clouds(gen_pcs, mode=args.normalize, logger=logger)
    #gen_pcs = 0.5 * torch.stack(gen_pcs, dim=0)

# Save
logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'out.npy'), gen_pcs.numpy())

# Compute metrics
with torch.no_grad():
    results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.batch_size)
    results = {k:v.item() for k, v in results.items()}
    jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results['jsd'] = jsd

for k, v in results.items():
    logger.info('%s: %.12f' % (k, v))
