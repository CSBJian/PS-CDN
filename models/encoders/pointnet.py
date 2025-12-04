import torch
import torch.nn.functional as F
from torch import nn


def cosine_distance(src, dst):
    """
    Calculate Spherical distance between each two points in sphere.
    任意两个向量的余弦距离  ----  球面空间就是 球面距离
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, C = src.shape
    _, M, _ = dst.shape
    src = src.unsqueeze(2)
    src = src.expand(B, N, M, C)
    dst = dst.unsqueeze(1)
    dst = dst.expand(B, N, M, C)
    src = torch.flatten(src, 0, 2)
    dst = torch.flatten(dst, 0, 2)
    inner = torch.sum(src * dst, dim=1, keepdim=True)
    src_norm = torch.sqrt(torch.sum(src ** 2, dim=1, keepdim=True))
    dst_norm = torch.sqrt(torch.sum(dst ** 2, dim=1, keepdim=True))
    cos_theta = inner / (src_norm * dst_norm)
    pairwise_distance = 1 - cos_theta
    # cos_theta   [-1  1]   值越小，角度距离越远，角度越大，值越接近1，角趋向于0
    # 1-cos_theta   [0 2]   值越小，角趋向于0，角度越小，值越接近1，角度越大   转换成正相关
    pairwise_distance = pairwise_distance.reshape([B, N, M])
    return pairwise_distance


def angle_density(xyz, npoints = 32):
    '''
    feat: input points feature data, [B, N, C]
    '''
    # import ipdb; ipdb.set_trace()
    sqrdists = cosine_distance(xyz, xyz)
    group_value, group_idx = torch.topk(sqrdists, npoints, dim=-1, largest=False, sorted=False)
    feat_density = group_value.mean(dim=-1)
    inverse_density = 1.0 / feat_density
    inverse_density = inverse_density / torch.max(inverse_density)   # 归一化到0-1之内了
    return inverse_density


class DensityAttn(nn.Module):
    def __init__(self, out_dim):
        super(DensityAttn, self).__init__()
        self.q_conv = nn.Conv1d(1, out_dim // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(1, out_dim // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.softmax = nn.Softmax(dim=-1)
        self.trans_conv = nn.Conv1d(out_dim, out_dim, 1)
        self.after_norm = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()

    def forward(self, x, density):
        x_q = self.q_conv(density).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(density)
        # b, n, n
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v


class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v


class VMFPointNetEncoder(nn.Module):
    def __init__(self, args):      # batch * 3 * 2048   ===>  batch * u  batch * k
        super().__init__()
        self.args = args
        self.zdim = args.latent_dim
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, self.zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, 1)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        u = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        u = F.relu(self.fc_bn2_m(self.fc2_m(u)))
        u = self.fc3_m(u)
        u = u / u.norm(dim=-1, keepdim=True)

        k = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        k = F.relu(self.fc_bn2_v(self.fc2_v(k)))
        k = self.fc3_v(k)
        k = F.softplus(k)+1

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return u, k


class PointNet_main(nn.Module):
    def __init__(self, num_classes=40, embedding=1024):
        super(PointNet_main, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, embedding, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embedding)
        self.linear1 = nn.Linear(embedding, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, num_classes)
        self.DensityAttn1 = DensityAttn(64)
        self.DensityAttn2 = DensityAttn(64)
        self.DensityAttn3 = DensityAttn(64)
        self.DensityAttn4 = DensityAttn(128)

    def forward(self, x):
        angledensity = angle_density(x.permute(0, 2, 1), 32).unsqueeze(2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.DensityAttn1(x,angledensity.permute(0, 2, 1))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.DensityAttn2(x, angledensity.permute(0, 2, 1))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.DensityAttn3(x, angledensity.permute(0, 2, 1))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.DensityAttn4(x, angledensity.permute(0, 2, 1))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class VMFPointNetDensityEncoder(nn.Module):
    def __init__(self, args):      # batch * 3 * 2048   ===>  batch * u  batch * k
        super().__init__()
        self.args = args
        self.zdim = args.latent_dim
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.DensityAttn1 = DensityAttn(128)
        self.DensityAttn2 = DensityAttn(128)
        self.DensityAttn3 = DensityAttn(256)
        self.DensityAttn4 = DensityAttn(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, self.zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, 1)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        angledensity = angle_density(x.permute(0, 2, 1), 32).unsqueeze(2)
        self.angledensity = angledensity
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.DensityAttn1(x,angledensity.permute(0, 2, 1))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.DensityAttn2(x,angledensity.permute(0, 2, 1))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.DensityAttn3(x,angledensity.permute(0, 2, 1))
        x = self.bn4(self.conv4(x))
        x = self.DensityAttn4(x,angledensity.permute(0, 2, 1))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        u = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        u = F.relu(self.fc_bn2_m(self.fc2_m(u)))
        u = self.fc3_m(u)
        u = u / u.norm(dim=-1, keepdim=True)

        k = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        k = F.relu(self.fc_bn2_v(self.fc2_v(k)))
        k = self.fc3_v(k)
        k = F.softplus(k)+1

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return u, k

    def density(self):
        return self.angledensity


