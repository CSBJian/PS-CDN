# -*- encoding: utf-8 -*-
# @ModuleName: func
# @Author: BiJian
# @Time: 2023/3/2 16:06
import numpy as np
import torch


#三维坐标转球面
def xyz2Spherical(point):
    x = point[:, :, 0].numpy()
    y = point[:, :, 1].numpy()
    z = point[:, :, 2].numpy()
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = torch.tensor(azimuth).unsqueeze(2)
    elevation = torch.tensor(elevation).unsqueeze(2)
    r = torch.tensor(r).unsqueeze(2)
    angle = torch.cat([azimuth, elevation], dim=2)
    return angle, r


#球面转三维坐标
def Spherical2xyz(angle, r):
    device = angle.device
    angle = angle.to('cpu')
    r = r.to('cpu')
    r = r.squeeze(2)

    azimuth = angle[:, :, 0].numpy()
    elevation = angle[:, :, 1].numpy()
    r = r.numpy()
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    x = torch.tensor(x).unsqueeze(2)
    y = torch.tensor(y).unsqueeze(2)
    z = torch.tensor(z).unsqueeze(2)
    point = torch.cat([x, y, z], dim=2)

    point = point.to(device)
    return point