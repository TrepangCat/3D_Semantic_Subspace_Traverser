import argparse
import copy
import glob
import math
import os
import re
import sys
from functools import partial
from multiprocessing import Pool
from random import sample
from typing import Optional

import mcubes  # 这个应该是来自 pymcubes 包
import numpy as np
import pyrender
import scipy.io as mat_io
import torch
import trimesh
from easydict import EasyDict
from pyrender.offscreen import RenderFlags
from skimage import io
# from model import Generator, Discriminator, ShapeAE128
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from trimesh.exchange.obj import export_obj

from loss import get_adversarial_losses, get_regularizer

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# --------------------------------------------------------

'''
WRW note:
this code is originally named as train_GAN_204.py.
'''


class Dataset:
    def __init__(self, root, num_samples=20000, in_memory=False):
        self.root = root
        self.num_samples = num_samples
        self.in_memory = in_memory

        tmp_data_list = glob.glob(os.path.join(self.root, f'**/*.mat'), recursive=True)
        tmp_data_list = sorted(tmp_data_list)

        if self.in_memory:
            # self.data_list = [torch.load(x, map_location='cpu') for x in tqdm(tmp_data_list)]
            self.data_list = [mat_io.loadmat(x).get('shape_code') for x in tqdm(tmp_data_list)]
        else:
            self.data_list = tmp_data_list

    def __getitem__(self, index):
        if self.in_memory:
            data = self.data_list[index]
        else:
            # data = torch.load(self.data_list[index], map_location='cpu')
            data = mat_io.loadmat(self.data_list[index]).get('shape_code')

        if type(data) == np.ndarray:
            data = torch.Tensor(data)

        return data

    def __len__(self):
        return len(self.data_list)


def get_activation(activation: str = "lrelu"):
    actv_layers = {
        "relu": nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, 0.2),
    }
    assert activation in actv_layers, f"activation [{activation}] not implemented"
    return actv_layers[activation]


def get_normalization(normalization: str = "batch_norm"):
    # 这个地方在二维网络改成三维网络时，没做修改。等报错再说吧。
    norm_layers = {
        "instance_norm": nn.InstanceNorm3d,
        "batch_norm": nn.BatchNorm3d,
        "group_norm": partial(nn.GroupNorm, num_groups=8),
        "layer_norm": partial(nn.GroupNorm, num_groups=1),
    }
    assert normalization in norm_layers, f"normalization [{normalization}] not implemented"
    return norm_layers[normalization]


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = 1,
            padding_mode: str = "zeros",
            groups: int = 1,
            bias: bool = True,
            transposed: bool = False,
            normalization: Optional[str] = None,
            activation: Optional[str] = "lrelu",
            pre_activate: bool = False,
            output_padding: int = None,
    ):
        if transposed:
            output_padding = stride - 1 if output_padding is None else output_padding
            conv = partial(nn.ConvTranspose3d, output_padding=output_padding)
            padding_mode = "zeros"
        else:
            conv = nn.Conv3d

        layers = [
            conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                groups=groups,
                bias=bias,
            )
        ]

        norm_actv = []
        if normalization is not None:
            norm_actv.append(
                get_normalization(normalization)(
                    num_features=in_channels if pre_activate else out_channels
                )
            )
        if activation is not None:
            norm_actv.append(
                get_activation(activation)(inplace=True)
            )

        if pre_activate:
            layers = norm_actv + layers
        else:
            layers = layers + norm_actv

        super().__init__(
            *layers
        )


class SubspaceLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            n_basis: int,
    ):
        super().__init__()

        self.U = nn.Parameter(torch.empty(n_basis, dim))
        nn.init.orthogonal_(self.U)
        self.L = nn.Parameter(torch.FloatTensor([3 * i for i in range(n_basis, 0, -1)]))
        self.mu = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        # z.shape = [b, n_basis]

        return (self.L * z) @ self.U + self.mu


# class PartLayer(nn.Module):
#     def __init__(self,
#                  part_index,  # 整个空间被切分为3**3=27块区域，这个index就是不同区域的索引。
#                  in_channels,
#                  n_basis: int,
#                  resolution: int,
#                  core_size,
#                  ):
#         super(PartLayer, self).__init__()
#
#         self.check_type(part_index)
#
#         core_size = int(resolution / 2) if core_size is None else core_size
#         periphery_size = int((resolution - core_size) / 2)
#         position_list = np.array([0, periphery_size, periphery_size + core_size], dtype=np.int32)
#         self.edge_list = np.array([periphery_size, core_size], dtype=np.int32)
#         self.type_list = np.array(['mini_cube', 'long_cuboid', 'square_cuboid', 'core_cube'])
#
#         self.indexes = part_index
#         self.part_num = len(part_index)
#         self.in_channels = in_channels
#         edges = self.edge_list[np.array(self.indexes) % 2]
#
#         self.id = []
#         self.jh = []
#         self.kw = []
#         for index in self.indexes:
#             position = [position_list[index]]
#
#             p_x, p_y, p_z = position
#
#             # 位置转换矩阵：
#             self.id = self.id.append(self.get_position_matrix(position=p_x, edge=i, resolution=resolution))
#
#             self.jh = self.get_position_matrix(position=p_y, edge=j, resolution=resolution)
#             self.kw = self.get_position_matrix(position=p_z, edge=k, resolution=resolution)
#
#         self.projection = SubspaceLayer(
#             dim=int((edges.prod() * self.part_num * self.in_channels).item()),
#             n_basis=n_basis,
#         )
#
#         self.i, self.j, self.k = edges
#         self.i = self.i * self.part_num
#
#     def get_position_matrix(self, position, edge, resolution):
#         matrix = torch.cat(
#             [torch.zeros(edge, position), torch.eye(edge), torch.zeros(edge, resolution - edge - position)], dim=1)\
#             .to(self.device)
#         return matrix
#
#     def check_type(self, index):
#         tmp = None
#         for i in index:
#             type = (np.array(i) % 2).sum()
#             if tmp == None:
#                 tmp = type
#             else:
#                 assert tmp == type
#
#         return True
#
#     def forward(self, z):
#         p = self.p
#
#         part_code = self.projection(z).view(-1, self.in_channels, self.i, self.j, self.k)
#         part_code = torch.einsum('bcijk,id,jh,kw->bcdhw', part_code, self.id, self.jh, self.kw)
#
#         return part_code


class SquarePartLayer(nn.Module):
    def __init__(self,
                 type,
                 resolution: int,
                 in_channels,
                 n_basis: int,
                 core_size=None,
                 device='cuda',
                 ):
        super(SquarePartLayer, self).__init__()
        self.axis = ['type_x', 'type_y', 'type_z'].index(type)

        self.index = [np.array([1, 1, 1]), np.array([1, 1, 1])]
        self.index[0][self.axis] = 0
        self.index[1][self.axis] = 2

        self.resolution = resolution
        self.in_channels = in_channels
        self.n_basis = n_basis
        self.device = device

        self.core_size = int(resolution / 2) if core_size is None else core_size
        periphery_size = int((resolution - self.core_size) / 2)
        position_list = np.array([0, periphery_size, periphery_size + self.core_size], dtype=np.int32)
        self.edge_list = np.array([periphery_size, self.core_size], dtype=np.int32)
        self.type = 'square_cuboid'
        # self.type_list = np.array(['mini_cube', 'long_cuboid', 'square_cuboid', 'core_cube'])

        edges = self.edge_list[np.array(self.index[0]) % 2]
        edge_x, edge_y, edge_z = edges

        # 获取位置转换矩阵：
        transform_matrix = [[], [], []]
        self.id, self.jh, self.kw = [], [], []

        idx = self.index[0]
        p = position_list[idx]
        transform_matrix[0].append(self.get_position_matrix(position=p[0], edge=edges[0], resolution=resolution))
        transform_matrix[1].append(self.get_position_matrix(position=p[1], edge=edges[1], resolution=resolution))
        transform_matrix[2].append(self.get_position_matrix(position=p[2], edge=edges[2], resolution=resolution))

        idx = self.index[1]
        p = position_list[idx]
        a = self.axis
        transform_matrix[a].append(self.get_position_matrix(position=p[a], edge=edges[a], resolution=resolution))

        self.id = torch.cat(transform_matrix[0])
        self.jh = torch.cat(transform_matrix[1])
        self.kw = torch.cat(transform_matrix[2])

        # 设置子空间
        self.i, self.j, self.k = edges * self.index[1]
        self.projection = SubspaceLayer(
            dim=int((edges.prod() * len(self.index) * self.in_channels).item()),
            n_basis=n_basis,
        )

    def get_position_matrix(self, position, edge, resolution):
        matrix = torch.cat(
            [torch.zeros(edge, position), torch.eye(edge), torch.zeros(edge, resolution - edge - position)], dim=1) \
            .to(self.device)
        return matrix

    def forward(self, z):
        # Get the part shape code.
        out = self.projection(z).view(-1, self.in_channels, self.i, self.j, self.k)

        # Enlarge the resolution of the shape code for differentiable embedding.
        out = torch.einsum('bcijk,id,jh,kw->bcdhw', out, self.id, self.jh, self.kw)

        return out


class CorePartLayer(nn.Module):
    def __init__(self,
                 resolution: int,
                 in_channels,
                 n_basis: int,
                 core_size=None,
                 device='cuda',
                 ):
        super(CorePartLayer, self).__init__()

        self.index = [1, 1, 1]
        self.resolution = resolution
        self.in_channels = in_channels
        self.n_basis = n_basis
        self.device = device

        self.core_size = int(resolution / 2) if core_size is None else core_size
        periphery_size = int((resolution - self.core_size) / 2)
        position_list = np.array([0, periphery_size, periphery_size + self.core_size], dtype=np.int32)
        self.edge_list = np.array([periphery_size, self.core_size], dtype=np.int32)
        self.type = 'core_cube'
        # self.type_list = np.array(['mini_cube', 'long_cuboid', 'square_cuboid', 'core_cube'])

        position = position_list[self.index]
        p_x, p_y, p_z = position

        edges = self.edge_list[np.array(self.index) % 2]
        self.i, self.j, self.k = edges

        # 位置转换矩阵：
        self.id = self.get_position_matrix(position=p_x, edge=self.i, resolution=resolution)
        self.jh = self.get_position_matrix(position=p_y, edge=self.j, resolution=resolution)
        self.kw = self.get_position_matrix(position=p_z, edge=self.k, resolution=resolution)

        self.projection = SubspaceLayer(
            dim=int((edges.prod() * self.in_channels).item()),
            n_basis=n_basis,
        )

    def get_position_matrix(self, position, edge, resolution):
        matrix = torch.cat(
            [torch.zeros(edge, position), torch.eye(edge), torch.zeros(edge, resolution - edge - position)], dim=1) \
            .to(self.device)
        return matrix

    def forward(self, z):
        # Get the part shape code.
        out = self.projection(z).view(-1, self.in_channels, self.i, self.j, self.k)

        # Enlarge the resolution of the shape code for differentiable embedding.
        out = torch.einsum('bcijk,id,jh,kw->bcdhw', out, self.id, self.jh, self.kw)

        return out


class ConstructionLayer(nn.Module):
    def __init__(self,
                 r: int,  # resolution
                 in_channels,
                 n_basis: int,
                 n_parts: int,
                 core_size=None,
                 device='cuda',
                 ):
        super(ConstructionLayer, self).__init__()
        assert n_parts in [4]  # 列表中的数目，代表着不同的子空间切分方式。

        self.n_basis = n_basis
        self.n_parts = n_parts
        self.part_list = nn.ModuleList()

        if self.n_parts == 4:
            # core_cube.
            self.part_list.append(
                CorePartLayer(
                    resolution=r,
                    in_channels=in_channels,
                    n_basis=n_basis,
                    core_size=core_size,
                    device=device,
                )
            )

            # square_cuboid
            for type in ['type_x', 'type_y', 'type_z']:
                self.part_list.append(
                    SquarePartLayer(
                        type=type,
                        resolution=r,
                        in_channels=in_channels,
                        n_basis=n_basis,
                        core_size=core_size,
                        device=device,
                    )
                )

        else:
            raise ValueError('Unknown \'n_parts\' value.')

    def forward(self, z_list):
        # z_list: [b, n_parts * n_biasis]
        getter = None  # 这里 getter 的名字来自世界上第一部合体机器人动画，盖塔机器人。Getter Robo
        for i in range(self.n_parts):
            _z = z_list[:, i * self.n_basis: (i + 1) * self.n_basis]
            part_code = self.part_list[i](_z)

            getter = getter + part_code if getter is not None else part_code

        return getter


class JigsawBlock(nn.Module):
    def __init__(self,
                 r: int,  # resolution
                 in_channels: int,
                 out_channels: int,
                 n_basis: int,
                 n_parts: int,
                 device='cuda',
                 ):
        super(JigsawBlock, self).__init__()

        self.assemble_parts = ConstructionLayer(
            r=r,
            in_channels=in_channels,
            n_basis=n_basis,
            n_parts=n_parts,
            core_size=None,
            device=device,
        )

        self.subspace_conv1 = ConvLayer(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            transposed=True,
            activation=None,
            normalization=None,
        )
        self.subspace_conv2 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            transposed=True,
            activation=None,
            normalization=None,
        )

        self.feature_conv1 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            transposed=True,
            pre_activate=True,
            normalization='batch_norm',
            # output_padding=0,  # 0066-GAN实验追加，与kernal_size=4一起使用
        )
        self.feature_conv2 = ConvLayer(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            transposed=True,
            pre_activate=True,
            normalization='batch_norm',
        )

    def forward(self, x, z):
        # z.shape = [b, n_basis]，取出的是当前模块的z

        phi = self.assemble_parts(z)
        out = self.feature_conv1(x + self.subspace_conv1(phi))
        out = self.feature_conv2(out + self.subspace_conv2(phi))

        return out


class Generator(nn.Module):
    def __init__(
            self,
            n_basis: int = 4,
            noise_dim: int = 256,
            code_channels: int = 128,
            max_channels: int = 512,
            device='cuda',
    ):
        super().__init__()

        self.n_basis = n_basis
        self.noise_dim = noise_dim
        self.n_parts = 4
        self.device = device

        self.layer_0 = nn.Sequential(
            ConvLayer(in_channels=self.noise_dim,
                      out_channels=128,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      normalization="batch_norm",
                      transposed=True,
                      output_padding=0,
                      pre_activate=False,
                      bias=False),
        )

        self.blocks = nn.ModuleList([
            JigsawBlock(
                r=4,
                in_channels=128,
                out_channels=64,
                n_basis=self.n_basis,
                n_parts=self.n_parts,
                device=device,
            ),  # in:4, out:8
            JigsawBlock(
                r=8,
                in_channels=64,
                out_channels=32,
                n_basis=self.n_basis,
                n_parts=self.n_parts,
                device=device,
            ),  # in:8, out:16
            # JigsawBlock(
            #     r=16,
            #     in_channels=32,
            #     out_channels=16,
            #     n_basis=self.n_basis,
            #     n_parts=self.n_parts,
            #     device=device,
            # ),  # in:16, out:32
        ])

        self.n_blocks = len(self.blocks)

        self.out_layer = nn.Sequential(
            nn.Conv3d(in_channels=32,
                      out_channels=code_channels,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1), ),
        )

    def sample_latents(self, batch: int, truncation=1.0):
        device = self.get_device()
        es = torch.randn(batch, self.noise_dim, device=device)
        zs = torch.randn(batch, self.n_blocks, self.n_basis * self.n_parts, device=device)
        # zs = torch.zeros(batch, self.n_blocks, self.n_basis * self.n_parts, device=device)

        if truncation < 1.0:
            es = torch.zeros_like(es) * (1 - truncation) + es * truncation
            zs = torch.zeros_like(zs) * (1 - truncation) + zs * truncation
        return es, zs

    def sample(self, batch: int, truncation=1.0, fix_z=None):
        _es, _zs = self.sample_latents(batch, truncation=truncation)
        out = self.forward(_es, _zs)
        return out

    def forward(self, es: torch.Tensor, zs: torch.Tensor):
        # input.shape = [b, 512]
        # zs.shape = [b, n_subspace_blocks, n_basis]

        out = self.layer_0(es[:, :, None, None, None])

        for block, z in zip(self.blocks, zs.permute(1, 0, 2)):
            # z.shape = [b, n_basis]，取出的是当前模块的z
            out = block(out, z)

        out = self.out_layer(out)
        return out

    def orthogonal_regularizer(self):
        reg = []
        for layer in self.modules():
            if isinstance(layer, SubspaceLayer):
                UUT = layer.U @ layer.U.t()
                reg.append(
                    ((UUT - torch.eye(UUT.shape[0], device=UUT.device)) ** 2).mean()
                )
        return sum(reg) / len(reg)

    def get_device(self):
        return self.device


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, normalization='batch'):
        super(self.__class__, self).__init__()

        self.num_features = in_channel
        self.res = False
        if stride is not 1:
            self.downsample = True
        else:
            self.downsample = False

        if normalization == 'spectral':
            self.conv = nn.Sequential(
                torch.nn.utils.spectral_norm(nn.Conv3d(in_channel, out_channel, kernel_size, stride, 1),
                                             n_power_iterations=5),
                nn.LeakyReLU(),
                torch.nn.utils.spectral_norm(nn.Conv3d(out_channel, out_channel, 3, 1, 1),
                                             n_power_iterations=5))
            if in_channel is not out_channel:
                self.res = True
                self.residual = torch.nn.utils.spectral_norm(nn.Conv3d(in_channel, out_channel, 1, 1, 0),
                                                             n_power_iterations=5)
        elif normalization == 'batch':
            self.conv = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size, stride, 1, bias=False),
                nn.BatchNorm3d(out_channel),
                nn.LeakyReLU(),
                nn.Conv3d(out_channel, out_channel, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_channel))
            if in_channel is not out_channel:
                self.res = True
                self.residual = nn.Sequential(
                    nn.Conv3d(in_channel, out_channel, 1, 1, 0, bias=False),
                    nn.BatchNorm3d(out_channel))
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size, stride, 1),
                nn.LeakyReLU(),
                nn.Conv3d(out_channel, out_channel, 3, 1, 1))
            if in_channel is not out_channel:
                self.res = True
                self.residual = nn.Conv3d(in_channel, out_channel, 1, 1, 0)

    def forward(self, x):
        out = self.conv(x)

        if self.downsample:
            res = F.max_pool3d(x, kernel_size=2)
            if self.res:
                res = self.residual(res)
        else:
            if self.res:
                res = self.residual(x)
            else:
                res = x
        out = out + res

        return F.leaky_relu(out)


class DiscriminatorPatchResidual(nn.Module):
    def __init__(self, code_channels=32, resolution=32, fc=False):
        super().__init__()
        self.resolution = resolution
        self.fc = fc

        self.main = nn.Sequential(
            ResidualBlock(code_channels, 32, 3, 1, normalization='spectral'),
            ResidualBlock(32, 64, 4, 2, normalization='spectral'),
            ResidualBlock(64, 128, 4, 2, normalization='spectral'),
            torch.nn.utils.spectral_norm(nn.Conv3d(128, 1, 1, 1, 0), n_power_iterations=5)
        )

        if self.fc:
            h = int(resolution / 4)
            self.classifier = nn.Sequential(
                nn.Linear((h ** 3) * 1, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1)
            )

    def forward(self, x):
        x = F.interpolate(x, size=self.resolution, mode='trilinear', align_corners=False)  # 这个是原版
        # x = F.interpolate(x, size=self.resolution, mode='trilinear', align_corners=True)

        output = self.main(x)
        # output = output.squeeze()
        if self.fc == True:
            output_one = self.classifier(output.view(len(output), -1))
            return output, output_one
        else:
            return output


from kornia.filters import filter3d


class Blur3d(nn.Module):
    def __init__(self, dowm=None):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, None, :] * \
            f[None, None, :, None] * f[None, :, None, None]
        return filter3d(x, f, normalized=True)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


def exists(val):
    return val is not None


def leaky_relu(p=0.2, ):
    return nn.LeakyReLU(p, inplace=True)


class vol_DiscriminatorBlock_v2(nn.Module):
    def __init__(self, input_channels, filters, k=3, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv3d(input_channels, filters, 1,
                                  stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv3d(input_channels, filters, k, padding=1),
            # nn.BatchNorm3d(filters),
            leaky_relu(),
            nn.Conv3d(filters, filters, 3, padding=1),
            # nn.BatchNorm3d(filters),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur3d(),
            nn.Conv3d(filters, filters, 3, padding=1, stride=2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        return (x + res) * (1 / math.sqrt(2))


# class vol_DiscriminatorBlock_spectral(nn.Module):
#     def __init__(self, input_channels, filters, downsample=True):
#         super().__init__()
#         self.conv_res = nn.Conv3d(input_channels, filters, 1,
#                                   stride=(2 if downsample else 1))
# 
#         self.net = nn.Sequential(
#             torch.nn.utils.spectral_norm(
#                 nn.Conv3d(input_channels, filters, 3, padding=1),
#                 n_power_iterations=5),
#             leaky_relu(),
#             torch.nn.utils.spectral_norm(
#                 nn.Conv3d(filters, filters, 3, padding=1),
#                 n_power_iterations=5),
#             leaky_relu()
#         )
# 
#         self.downsample = nn.Sequential(
#             Blur3d(),
#             torch.nn.utils.spectral_norm(
#                 nn.Conv3d(filters, filters, 3, padding=1, stride=2),
#                 n_power_iterations=5),
#         ) if downsample else None
# 
#     def forward(self, x):
#         res = self.conv_res(x)
#         x = self.net(x)
#         if exists(self.downsample):
#             x = self.downsample(x)
#         return (x + res) * (1 / math.sqrt(2))


from math import log2


class Discriminator_stylegan_patchgan(nn.Module):
    def __init__(self, size, max_channels=256, network_capacity=16, code_channels=4):
        super().__init__()
        num_layers = int(log2(size) - 1) + 1  # 这里的1是为了使用构建一个通道转换的layer
        self.volume_size = size
        num_init_filters = code_channels
        filters = [num_init_filters] + \
                  [network_capacity * num_init_filters * (4 ** i)
                   for i in range(num_layers + 1)]

        set_fmap_max = partial(min, max_channels)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        for _, (in_chan, out_chan) in enumerate(chan_in_out):
            block = vol_DiscriminatorBlock_v2(in_chan, out_chan, k=4 if len(blocks) > 0 else 3)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        # self.flatten = Flatten()
        # self.to_logit = nn.Linear(filters[-1], 1)
        self.out_conv = nn.Conv3d(out_chan, 1, 3, stride=1, padding=1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        # x = self.flatten(x)
        # x = self.to_logit(x)
        # return x.squeeze()

        x = self.out_conv(x)
        return x


def save_mesh_from_points_for_pool(input):
    coord_occ = input['coord_occ']
    save_dir = input['save_dir']
    index = input['index']
    no_png = input['no_png']
    seed = input['seed']
    resolution = input['resolution']
    threshold = input['threshold'] if 'threshold' in input.keys() else 0.0

    save_mesh_from_points(coord_occ=coord_occ,
                          save_dir=save_dir,
                          index=index,
                          no_png=no_png,
                          seed=seed,
                          resolution=resolution,
                          threshold=threshold,
                          )


def save_mesh_from_points(coord_occ, save_dir, index, no_png=False, seed=-1, resolution=256, threshold=0.5):
    assert coord_occ.ndim == 1
    assert isinstance(coord_occ, torch.Tensor) or isinstance(coord_occ, np.ndarray)

    if isinstance(coord_occ, torch.Tensor):
        coord_occ = coord_occ.numpy()

    dir = save_dir
    if not os.path.exists(dir):
        os.makedirs(dir)

    dir_name = os.path.basename(save_dir)
    name_obj = os.path.join(dir, f'sdfmesh_{dir_name}-seed{seed}-index{index}.obj')

    mesh = mesh_from_logits(coord_occ, resolution, threshold)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return
    _write_mesh_as_obj(mesh=mesh, output_path=name_obj)

    if not no_png:
        img_path = name_obj[0:-4] + '.png'
        from_trimesh_to_img(mesh, img_path, rotate_x=20, rotate_y=-120, rewrite=False)  # 用于 shapenetV1 的可视化
        # from_trimesh_to_img(mesh, img_path, rotate_x=15, rotate_y=150, rewrite=False)  # 用于 shapenetV2 的可视化


def _write_mesh_as_obj(mesh: trimesh.Trimesh, output_path: str):
    # Can include texture, see in the parameters of export_obj()
    if '.obj' not in os.path.basename(output_path):
        raise ValueError('output_path does not contain .obj')

    # If, the .obj with the same name is there, delete it.
    if os.path.exists(output_path):
        os.remove(output_path)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    obj = export_obj(mesh=mesh, include_normals=True, include_color=True)
    with open(output_path, 'w') as _f:
        _f.write(obj)


# ----------------------------------------------------------------------------
# 从 AE 的 decoder 的输出中，恢复得到 mesh
def mesh_from_logits(logits, resolution=256, t=0.5):
    # 下面这些参数，要和前面的 point_coords 的参数一致
    resolution = resolution
    _threshold = t
    max = 0.5
    min = -0.5

    logits = np.reshape(logits, (resolution,) * 3)

    # padding to ba able to retrieve object close to bounding box bondary
    logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=-10.0)
    # threshold = np.log(_threshold) - np.log(1. - _threshold)
    threshold = _threshold
    vertices, triangles = mcubes.marching_cubes(logits, threshold)

    # remove translation due to padding
    vertices -= 1

    # rescale to original scale
    step = (max - min) / (resolution - 1)
    vertices = np.multiply(vertices, step)
    vertices += [min, min, min]

    return trimesh.Trimesh(vertices, triangles)


def from_trimesh_to_img(mesh: trimesh.Trimesh, img_path: str, color=[128, 128, 128],
                        rotate_x=0, rotate_y=0, rewrite=False):
    if os.path.exists(img_path) and not rewrite:
        # print('image exists')
        return

    def rotationx(theta):
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi), 0.0],
            [0.0, -np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    def rotationy(theta):
        return np.array([
            [np.cos(theta / 180 * np.pi), 0.0, np.sin(theta / 180 * np.pi), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(theta / 180 * np.pi), 0.0, np.cos(theta / 180 * np.pi), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    # Rendering parameters
    ambient_light = 0.8
    directional_light = 1.0

    # Create a scene
    scene = pyrender.Scene(ambient_light=np.array([ambient_light, ambient_light, ambient_light]))

    v, f = mesh.vertices, mesh.faces
    # Note that sometimes normalization is not needed.
    # vox_mesh = trimesh.Trimesh(vertices=v, vertex_colors=[255, 140, 0], faces=f)
    # v = (v + 0.5) / 128.0 - 1.0  # Normalization, [-0.5, 255.5] => [-1.0, 1.0]
    v = (v - v.min()) / (v.max() - v.min()) * 2.0 - 1.0  # Normalization, [min, max] => [-1.0, 1.0]
    vox_mesh = trimesh.Trimesh(vertices=v, face_colors=color, faces=f)
    pyrender_mesh = pyrender.Mesh.from_trimesh(vox_mesh, wireframe=False, smooth=False)
    mesh_node = scene.add(pyrender_mesh, pose=np.matmul(rotationy(rotate_y), rotationx(rotate_x)))

    # Create a camera
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, znear=0.05, aspectRatio=1.0)
    cam_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])  # If you can't see the whole mesh, there must be something wrong with the camera pose.
    scene.add(cam, pose=cam_pose)

    # Create directional lights
    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_1 = scene.add(direc_l, pose=np.matmul(rotationy(30), rotationx(45)))

    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_2 = scene.add(direc_l, pose=np.matmul(rotationy(-30), rotationx(45)))

    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_3 = scene.add(direc_l, pose=np.matmul(rotationy(-180), rotationx(45)))

    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=(directional_light - 0.5))
    light_node_4 = scene.add(direc_l, pose=np.matmul(rotationy(0), rotationx(-10)))

    '''
    Rendering
    '''
    render_flags = {
        'flip_wireframe': False,
        'all_wireframe': False,
        'all_solid': False,
        'shadows': True,
        'vertex_normals': False,
        'face_normals': False,
        'cull_faces': True,
        'point_size': 1.0,
    }
    viewer_flags = {
        'mouse_pressed': False,
        'rotate': False,
        'rotate_rate': np.pi / 2.0,
        'rotate_axis': np.array([18.0, 27.0, 0]),
        'view_center': np.array([0.0, 0.0, 0.0]),
        'record': False,
        'use_raymond_lighting': False,
        'use_direct_lighting': True,
        'lighting_intensity': 3.0,
        'use_perspective_cam': False,
        'window_title': 'undefined',
        'refresh_rate': 25.0,
        'fullscreen': False,
        'show_world_axis': False,
        'show_mesh_axes': False,
        'caption': None,
        'save_one_frame': False,
    }

    # v = pyrender.Viewer(scene, viewport_size=(800, 800), render_flags=render_flags,
    #                     viewer_flags=viewer_flags, run_in_thread=False)
    # v.close()

    r = pyrender.OffscreenRenderer(viewport_width=2000, viewport_height=2000)
    # flags = RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.OFFSCREEN | RenderFlags.ALL_WIREFRAME
    flags = RenderFlags.SHADOWS_ALL | RenderFlags.OFFSCREEN
    color, depth = r.render(scene, flags=flags)  # color=[R, G, B]
    r.delete()

    io.imsave(fname=img_path, arr=color)

    if not os.path.exists(img_path):
        raise IOError('image not exists')

    return color


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


@torch.no_grad()
def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]

    print()
    s = '\n'
    for row in rows:
        tmp = '  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths))
        print(tmp)
        s = s + tmp + '\n'
    print()
    s += '\n'
    return outputs, s


# to extend the bool input in args.
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_voxel_coordinates(resolution_coord=32, min_coord=-1.0, max_coord=1.0, center=[0, 0, 0]):
    assert len(center) == 3
    if not type(center) == torch.Tensor:
        center = torch.tensor(center, dtype=torch.float32, device='cuda')

    # Define the point_coords
    grid_points = create_grid_points_from_bounds(min_coord, max_coord, resolution_coord)
    grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()  # 为什么要写这一行？——交换x轴和z轴的位置，使得可视化正常。
    # 这里可以用 np.swapaxes 代替。当然得把尺寸修改下

    # a = max_coord + min_coord
    # b = max_coord - min_coord
    #
    # # 改变 min_coord, max_coord 的设定，下面这一行可以是多余的，之后的版本里可以考虑删去
    # # grid_coords = 2 * grid_points - a  # 将值域从 [-0.5, 0.5] 变换到 [-1.0, 1.0]
    # # grid_coords = grid_coords / b

    grid_points = torch.tensor(grid_points, dtype=torch.float32, device='cuda') + center

    return grid_points


# def vaild_mask(mask):
#     local_patch_number = 16  # 这里因为代码弃用，所以并没有使用这里的 local_patch_number
#     for i in range(len(mask)):  # 这里其实就是按照batch_size再确认数目
#         if torch.count_nonzero(mask[i]) < local_patch_number:
#             print(f"mask_{i} number:", torch.count_nonzero(mask[i]))
#             return False
#     return True


def get_patch(pred_occ, shape_code, iso_value, local_patch_size=0.125, local_patch_number=16, resolution=16):
    # 获取潜在的点的顺序
    topk_occ = torch.topk(torch.abs(pred_occ - iso_value),
                          k=2048,
                          largest=False,
                          sorted=False)
    _grid_points = get_voxel_coordinates(32)[topk_occ.indices, :]  # 这个地方的32分辨率与外面的分辨率完全对应。

    # 准备好 mask 将选择的中心点限定在一定范围内
    # local_patch_size = 0.125
    mask = (_grid_points[:, :, 0] <= 1 - local_patch_size) & \
           (_grid_points[:, :, 0] >= - 1 + local_patch_size) & \
           (_grid_points[:, :, 1] <= 1 - local_patch_size) & \
           (_grid_points[:, :, 1] >= - 1 + local_patch_size) & \
           (_grid_points[:, :, 2] <= 1 - local_patch_size) & \
           (_grid_points[:, :, 2] >= - 1 + local_patch_size)
    # 这里通过函数确认mask中True的数目。这里我自己生成的结果可能就在边界上，所以会导致一些问题。关于下面这行代码是否使用，还有待讨论
    # SDF-StyleGAN中因为有padding，所以大体上不需要担心这个问题。
    # assert vaild_mask(mask)

    patch_centers = torch.zeros(len(shape_code),  # 这个地方可以替换为batch_size
                                local_patch_number,
                                3,
                                device='cuda')

    # 从备选项中，随机选取中心点。
    for i in range(len(shape_code)):
        candidate = _grid_points[i][mask[i]]
        index = torch.tensor(sample(range(candidate.shape[0]), local_patch_number),
                             dtype=torch.long,
                             device='cuda')
        patch_centers[i] = candidate[index]
    patch_centers = patch_centers.detach()

    # 根据中心点，构建对应的局部坐标。
    n = resolution ** 3
    tmp_local_voxel_coordinates = get_voxel_coordinates(resolution,
                                                        min_coord=-local_patch_size,
                                                        max_coord=local_patch_size, )
    patch_points = torch.zeros(len(shape_code), local_patch_number, n, 3, device='cuda')
    for i in range(len(shape_code)):
        for j in range(local_patch_number):
            patch_points[i, j] = tmp_local_voxel_coordinates + patch_centers[i, j]

    # 构建好局部坐标后，根据局部坐标，插值得到局部 shape_code
    _b = len(shape_code) * local_patch_number
    c = shape_code.shape[1]
    patch_points = patch_points.unsqueeze(1)  # [b, 1, local_patch_number, n**3, 3]
    local_shape_codes = F.grid_sample(shape_code, patch_points,
                                      align_corners=True)  # [b, c, 1, local_patch_number, n**3]

    # 这里记得将 local_shape_codes 的形状还原到16分辨率
    local_shape_codes = torch.swapaxes(local_shape_codes, 1, 3)
    local_shape_codes = local_shape_codes.reshape(_b, c, resolution, resolution, resolution)

    return local_shape_codes


# ============================
# Define the point_coords
min_coord = -0.5
max_coord = 0.5
resolution_coord = 128
grid_points = create_grid_points_from_bounds(min_coord, max_coord, resolution_coord)
grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()  # 为什么要写这一行？——交换x轴和z轴的位置，使得可视化正常。
# 这里可以用 np.swapaxes 代替。

a = max_coord + min_coord
b = max_coord - min_coord

grid_coords = 2 * grid_points - a  # 将值域从 [-0.5, 0.5] 变换到 [-1.0, 1.0]
grid_coords = grid_coords / b

grid_coords = torch.from_numpy(grid_coords).to('cuda', dtype=torch.float)
grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to('cuda')
grid_points_split = torch.split(grid_coords, 500000, dim=1)
# ============================

from train_VAE import ShapeAE64

# ae_ckpt_path = '/home/brl/data_disk_4t/code/stylegan_series/A_new_gan_project_july/run_CubeCodeCraft/0186-AE-batch16-steps200/checkpoints/step00000200.pt'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default='./sample_data/ShapeNetCore.v1_processed_tmp_encoded',
        help="path to the VAE encoded dataset",
    )
    parser.add_argument(
        "--ae_ckpt_path",
        type=str,
        default='./checkpoint/vae_ckpt-0186/step00000200.pt',
        help="path to the VAE ckpt",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20000,
        help="num of sample for dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu/cuda (does not support multi-GPU training for now)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="run_CubeCodeCraft",
        help="ourput directory",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=16,
        help="image size",
    )
    # parser.add_argument(  # abandoned
    #     "--local_size",
    #     type=int,
    #     default=8,
    #     help="image size",
    # )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="train steps",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=50,
        help="sample log period",
    )
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=50,
        help="checkpoint save period",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        # default=1e-3,  # grid im-gan 的参数
        help="base learning rate",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=10,
        help="num of samples for visualization",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=6,
        help="num of workers for dataloader",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint file to continue training",
    )
    parser.add_argument(
        "--n_basis",
        type=int,
        default=6,
        help="subspace dimension for a generator layer",
    )
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=512,
        help="noise dimension for the input layer of the generator",
    )
    # parser.add_argument(
    #     "--base_channels",
    #     type=int,
    #     default=16,
    #     help="num of base channels for generator/discriminator",
    # )
    parser.add_argument(
        "--code_channels",
        type=int,
        default=8,
        help="num of shape code channels for generator",
    )
    parser.add_argument(
        "--max_channels",
        type=int,
        default=256,
        help="max num of channels for generator/discriminator",
    )
    parser.add_argument(
        "--adv_loss",
        choices=["hinge", "non_saturating", "lsgan", "ns", "wgan"],
        default="non_saturating",
        help="adversarial loss type",
    )
    parser.add_argument(
        "--orth_reg",
        type=float,
        default=100.0,
        help="basis orthogonality regularization weight",
    )
    parser.add_argument(
        "--d_reg",
        type=float,
        default=10.0,
        help="discriminator r1 regularization weight",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=4,
        help="discriminator lazy regularization period",
    )
    parser.add_argument(
        "--train_ae",
        type=str2bool,
        default=True,
        help="whether to train AE",
    )
    parser.add_argument(
        "--train_gan",
        type=str2bool,
        default=False,
        help="whether to train GAN",
    )
    parser.add_argument(
        "--no_png",
        type=str2bool,
        default=False,
        help="whether to train GAN",
    )
    args = parser.parse_args()
    print(args)

    batch_size = args.batch
    device = args.device

    # 为了局部鉴别器而准备的grid坐标
    grid_points_32 = get_voxel_coordinates(32)
    grid_points_32 = grid_points_32.unsqueeze(0).repeat(batch_size, 1, 1)  # 修改尺寸和batch_size

    # Set the seed. Make the experiments repeatable.
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True

    # AE
    AE = ShapeAE64(
        gen_channels=args.code_channels,
    ).to(device).eval().requires_grad_(False)
    ae_ckpt_path = args.ae_ckpt_path
    ae_ckpt = torch.load(ae_ckpt_path)
    AE.load_state_dict(ae_ckpt["ae"])
    assert args.code_channels == ae_ckpt["args"].code_channels

    # models
    G = Generator(
        n_basis=args.n_basis,
        noise_dim=args.noise_dim,
        code_channels=args.code_channels,
        max_channels=args.max_channels,
        device=device
    ).to(device).train()
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)

    D = Discriminator_stylegan_patchgan(
        size=args.size / 4,
        # base_channels=args.base_channels,
        # base_channels=256,
        code_channels=args.code_channels,
        max_channels=args.max_channels,
    ).to(device).train()

    # optimizers
    g_optim_dict = {'whole': torch.optim.Adam(
        G.parameters(),
        lr=args.lr,
        betas=(0.0, 0.99),  # 这个是 grid im-gan 优化器的参数
        # betas=(0.5, 0.99),
    )}
    # # 下面是用于编辑的不同子空间的优化器。
    # g_part_optim_list: List[Adam] = []
    # for layer in G.modules():
    #     if isinstance(layer, SubspaceLayer):
    #         g_part_optim_list.append(
    #             torch.optim.Adam(
    #                 layer.parameters(),
    #                 lr=args.lr,
    #                 betas=(0.0, 0.99),  # 这个是 grid im-gan 优化器的参数
    #                 # betas=(0.5, 0.99),
    #             )
    #         )
    # g_optim_dict['parts_optim'] = g_part_optim_list

    d_optim = torch.optim.Adam(
        D.parameters(),
        lr=args.lr,
        betas=(0.0, 0.99),  # 这个是 grid im-gan 优化器的参数
        # betas=(0.5, 0.99),
    )

    start_step = 0
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        start_step = ckpt["step"]
        G.load_state_dict(ckpt["g"])
        G_ema.load_state_dict(ckpt["g_ema"])
        D.load_state_dict(ckpt["d"])
        g_optim_dict['whole'].load_state_dict(ckpt["g_whole_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

        # for i in range(len(g_optim_dict['parts_optim'])):
        #     g_optim_dict['parts_optim'][i].load_state_dict(ckpt[f'g_optim_part_{i}'])

    # losses
    d_adv_loss_fn, g_adv_loss_fn = get_adversarial_losses(args.adv_loss)
    d_reg_loss_fn = get_regularizer("r1")
    # d_reg_loss_fn = get_regularizer("gradient_regularization")

    # data
    dataset = Dataset(args.path, args.num_samples, in_memory=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.n_workers,
        prefetch_factor=32,
    )

    # train utils_sdfstylegan
    ema = partial(accumulate, decay=0.5 ** (args.batch / (10 * 1000)))
    # ema = partial(accumulate, decay=0.5)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(args.output_dir):
        prev_run_dirs = [x for x in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    is_resume = '-resumed' if args.ckpt is not None else ''
    run_dir = os.path.join(args.output_dir, f'{cur_run_id:04d}-GAN-batch{args.batch}-steps{args.steps}' + is_resume)
    assert not os.path.exists(run_dir)
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    tb_writer = SummaryWriter(run_dir)
    es_sample, zs_sample = G_ema.sample_latents(args.n_sample)
    print(f"training log directory: {run_dir}")

    first_visualization = True

    # print module summary
    with open(os.path.join(run_dir, "setting.txt"), 'w') as f:
        f.write(f'{sys.argv[0]}\n\n')

        example = next(iter(loader))
        # example_p = example.get('grid_coords').to(device)  # 散点的坐标
        # example_vox = example.get('vox').to(device)  # 体素

        es_tmp, zs_tmp = G_ema.sample_latents(args.batch)
        example_fake_code, str_g = print_module_summary(module=G, inputs=[es_tmp, zs_tmp])
        f.write(str_g)
        _, str_d = print_module_summary(module=D, inputs=[example_fake_code])
        f.write(str_d)

    # train loop
    # iterator = tqdm(range(args.steps), initial=start_step, ncols=120)
    d_loss_avg = 0
    d_local_loss_avg = 0
    r1_avg = 0
    r1_local_avg = 0
    g_loss_avg = 0
    g_local_loss_avg = 0
    g_loss_reg_avg = 0
    g_loss_adv_avg = 0
    # g_loss_parts_avg = 0
    # g_loss_reg_parts_avg = 0
    # g_loss_adv_parts_avg = 0

    d_loss = torch.Tensor([0]).to(device)
    d_local_loss = torch.Tensor([0]).to(device)
    r1 = torch.Tensor([0]).to(device)
    r1_local = torch.Tensor([0]).to(device)
    g_loss = torch.Tensor([0]).to(device)
    g_local_loss = torch.Tensor([0]).to(device)
    g_loss_reg = torch.Tensor([0]).to(device)
    g_loss_adv = torch.Tensor([0]).to(device)
    # g_loss_parts = torch.Tensor([0]).to(device)
    # g_loss_reg_parts = torch.Tensor([0]).to(device)
    # g_loss_adv_parts = torch.Tensor([0]).to(device)

    epoch_iter = tqdm(range(args.steps),
                      initial=start_step,
                      # ncols=80,
                      # ncols=100,
                      # position=0
                      )

    for step in epoch_iter:
        step = step + start_step + 1
        if step > args.steps:
            break

        iterator = tqdm(loader,
                        # ncols=80,
                        # ncols=100,
                        # position=0
                        )
        for i, real_data in enumerate(iterator):
            # p = real_data.get('grid_coords').to(device)  # 散点的坐标
            # occ = real_data.get('occupancies').to(device)  # 散点的内外情况
            # vox = real_data.get('vox').to(device)  # 体素

            # D
            AE.eval().requires_grad_(False)
            D.train().requires_grad_(True)

            with torch.no_grad():
                fake_code = G.sample(args.batch)
                # real_code, _ = AE.encoder(vox)
                real_code = real_data.cuda()
            real_pred = D(real_code)
            fake_pred = D(fake_code)

            # d_loss = d_adv_loss_fn(real_pred, fake_pred, soft_labels=True)
            d_loss = d_adv_loss_fn(real_pred, fake_pred)
            # d_loss = 0
            # for loss_order in range(len(real_pred)):
            #     d_loss_tmp = d_adv_loss_fn(real_pred[loss_order], fake_pred[loss_order])
            #     d_loss = d_loss + d_loss_tmp

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # 鉴别器的归一化Loss
            if i % args.d_reg_every == 0:
                r1 = 0

                real_code.requires_grad = True
                real_pred = D(real_code)

                real_norm = d_reg_loss_fn(real_pred, real_code) * args.d_reg
                # real_norm = 0
                # for loss_order in range(len(real_pred)):
                #     tmp = d_reg_loss_fn(real_pred[loss_order], real_code) * args.d_reg
                #     real_norm = real_norm + tmp

                d_optim.zero_grad()
                # real_norm.backward()
                real_norm.backward(retain_graph=True)
                d_optim.step()

                # fake_code.requires_grad = True
                # fake_pred = D(fake_code)
                #
                # fake_norm = d_reg_loss_fn(fake_pred, fake_code) * args.d_reg
                # # fake_norm = 0
                # # for loss_order in range(len(fake_pred)):
                # #     tmp = d_reg_loss_fn(fake_pred[loss_order], fake_code) * args.d_reg
                # #     fake_norm = fake_norm + tmp
                #
                # d_optim.zero_grad()
                # fake_norm.backward(retain_graph=True)
                # d_optim.step()

                r1 = real_norm
                # r1 = real_norm + fake_norm

            # G
            G.train().requires_grad_(True)

            # 整体优化
            fake_code = G.sample(args.batch)
            fake_pred = D(fake_code)

            g_loss_adv = g_adv_loss_fn(fake_pred)
            # g_loss_adv = 0
            # for loss_order in range(len(fake_pred)):
            #     loss_tmp = g_adv_loss_fn(fake_pred[loss_order])
            #     g_loss_adv = g_loss_adv + loss_tmp

            g_loss_reg = G.orthogonal_regularizer() * args.orth_reg
            g_loss = g_loss_adv + g_loss_reg + g_local_loss
            g_optim_dict['whole'].zero_grad()
            # 部位优化
            # for i, opt in enumerate(g_optim_dict['parts_optim']):
            #     opt.zero_grad()

            g_loss.backward()

            g_optim_dict['whole'].step()
            # for i, opt in enumerate(g_optim_dict['parts_optim']):
            #     opt.step()

            ema(G_ema, G)

            # log
            iterator.set_description(
                f"d={d_loss.item():.4f}, d_local_loss={d_local_loss.item():.4f}, "
                f"d_r1={r1.item():.4f}, d_r1_local={r1_local.item():.4f}; "
                f"g:{g_loss.item():.4f}, g_local={g_local_loss.item():.4f}"
            )

            len_loader = len(loader)
            d_local_loss_avg += d_local_loss.item() / len_loader
            d_loss_avg += d_loss.item() / len_loader
            r1_avg += r1.item() / len_loader
            r1_local_avg += r1_local / len_loader
            g_loss_avg += g_loss.item() / len_loader
            g_local_loss_avg += g_local_loss.item() / len_loader
            g_loss_reg_avg += g_loss_reg.item() / len_loader
            g_loss_adv_avg += g_loss_adv.item() / len_loader

        tb_writer.add_scalar("loss/D", d_loss_avg, step)
        tb_writer.add_scalar("loss/D_r1", r1_avg, step)
        tb_writer.add_scalar("loss/D_local", d_local_loss_avg, step)
        tb_writer.add_scalar("loss/D_r1_local", r1_local_avg, step)
        tb_writer.add_scalar("loss/G", g_loss_avg, step)
        tb_writer.add_scalar("loss/G_local", g_local_loss, step)
        tb_writer.add_scalar("loss/G_orth", g_loss_reg_avg, step)
        tb_writer.add_scalar("loss/G_adv", g_loss_adv_avg, step)
        # tb_writer.add_scalar("loss/G_parts", g_loss_parts_avg, step)
        # tb_writer.add_scalar("loss/G_parts_orth", g_loss_reg_parts_avg, step)
        # tb_writer.add_scalar("loss/G_parts_adv", g_loss_adv_parts_avg, step)
        # tb_writer.add_scalar("loss/AE", ae_loss_avg, step)

        d_loss_avg = 0
        d_local_loss_avg = 0
        r1_avg = 0
        r1_local_avg = 0
        g_loss_avg = 0
        g_loss_reg_avg = 0
        g_loss_adv_avg = 0
        g_local_loss_avg = 0

        done = args.steps == step
        if step % args.sample_every == 1 or done or args.sample_every == 1:
            with torch.no_grad():
                G_ema.eval().requires_grad_(False)
                AE.eval().requires_grad_(False)

                if first_visualization:
                    # 可视化AE结果：
                    ae_visual_list = [(dataset[0]).unsqueeze(0).to(device),
                                      (dataset[1]).unsqueeze(0).to(device),
                                      (dataset[2]).unsqueeze(0).to(device),
                                      (dataset[3]).unsqueeze(0).to(device),
                                      (dataset[4]).unsqueeze(0).to(device),
                                      ]

                    for i in range(min(4, len(real_code))):
                        ae_visual_list += [real_code.split(1)[i]]  # 取出一部分数据。（这部分是随机的）

                    mission_list = []

                    index = 0
                    for shape_code in ae_visual_list:
                        shape_code = shape_code.to(device)

                        pred_occ_list = []
                        for points in grid_points_split:
                            pred_occ = AE.decoder(shape_code, points).squeeze(0).detach().cpu()
                            pred_occ_list.append(pred_occ)
                        all_pred_occ = torch.cat(pred_occ_list, dim=0)

                        save_dir = os.path.join(run_dir, f'real_AE_{step:05d}')

                        index += 1

                        input_dict = {'coord_occ': all_pred_occ,
                                      'save_dir': save_dir,
                                      'index': index,
                                      'no_png': args.no_png,
                                      'seed': seed,
                                      'resolution': resolution_coord,
                                      }

                        mission_list.append(input_dict)

                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        # save_mesh_from_points(coord_occ=all_pred_occ,
                        #                       save_dir=save_dir,
                        #                       index=index,
                        #                       no_png=args.no_png,
                        #                       )

                    with Pool(4) as p:
                        res_list = p.imap_unordered(save_mesh_from_points_for_pool, mission_list)
                        bar = tqdm(total=len(mission_list), desc='AE_visualization')
                        for res in res_list:
                            bar.update()

                    first_visualization = False

                # ===============================================
                # es_sample, zs_sample = log_sample
                fake_code = G(es_sample, zs_sample)
                # fake_code = G_ema(es_sample, zs_sample)

                mission_list = []

                for index in range(len(fake_code)):
                    shape_code = fake_code[index].unsqueeze(0)

                    pred_occ_list = []
                    for points in grid_points_split:
                        pred_occ = AE.decoder(shape_code, points).squeeze(0).detach().cpu()
                        pred_occ_list.append(pred_occ)
                    all_pred_occ = torch.cat(pred_occ_list, dim=0)

                    save_dir = os.path.join(run_dir, f'fake_G_{step:05d}')
                    # save_dir = os.path.join(run_dir, f'fake_Gema_{step:05d}')

                    input_dict = {'coord_occ': all_pred_occ,
                                  'save_dir': save_dir,
                                  'index': index,
                                  'no_png': args.no_png,
                                  'seed': seed,
                                  'resolution': resolution_coord,
                                  }

                    mission_list.append(input_dict)

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # save_mesh_from_points(coord_occ=all_pred_occ,
                    #                       save_dir=save_dir,
                    #                       index=index,
                    #                       no_png=args.no_png,
                    #                       )

                with Pool(4) as p:
                    res_list = p.imap_unordered(save_mesh_from_points_for_pool, mission_list)
                    bar = tqdm(total=len(mission_list), desc='GAN_visualization')
                    for res in res_list:
                        bar.update()

                # # ===============================================
                # # es_sample, zs_sample = log_sample
                # # fake_code = G(es_sample, zs_sample)
                # fake_code = G_ema(es_sample, zs_sample)
                # 
                # mission_list = []
                # 
                # for index in range(len(fake_code)):
                #     shape_code = fake_code[index].unsqueeze(0)
                # 
                #     pred_occ_list = []
                #     for points in grid_points_split:
                #         pred_occ = AE.decoder(shape_code, points).squeeze(0).detach().cpu()
                #         pred_occ_list.append(pred_occ)
                #     all_pred_occ = torch.cat(pred_occ_list, dim=0)
                # 
                #     save_dir = os.path.join(run_dir, f'fake_Gema_{step:05d}')
                #     # save_dir = os.path.join(run_dir, f'fake_Gema_{step:05d}')
                # 
                #     input_dict = {'coord_occ': all_pred_occ,
                #                   'save_dir': save_dir,
                #                   'index': index,
                #                   'no_png': args.no_png,
                #                   'seed': seed,
                #                   'resolution': resolution_coord,
                #                   }
                # 
                #     mission_list.append(input_dict)
                # 
                #     if not os.path.exists(save_dir):
                #         os.makedirs(save_dir)
                # 
                #     # save_mesh_from_points(coord_occ=all_pred_occ,
                #     #                       save_dir=save_dir,
                #     #                       index=index,
                #     #                       no_png=args.no_png,
                #     #                       )
                # 
                # with Pool(4) as p:
                #     res_list = p.imap_unordered(save_mesh_from_points_for_pool, mission_list)
                #     bar = tqdm(total=len(mission_list), desc='GAN_visualization')
                #     for res in res_list:
                #         bar.update()

        if step % args.ckpt_every == 1 or done or args.ckpt_every == 1:
            if not os.path.exists(os.path.join(run_dir, "checkpoints")):
                os.makedirs(os.path.join(run_dir, "checkpoints"))

            ckpt = {
                "step": step,
                "args": args,
                "g": G.eval().requires_grad_(False).state_dict(),
                "g_ema": G_ema.eval().requires_grad_(False).state_dict(),
                "d": D.eval().requires_grad_(False).state_dict(),
                "g_whole_optim": g_optim_dict['whole'].state_dict(),
                "d_optim": d_optim.state_dict(),
            }
            # # 把不同语义part对应的优化器保存
            # for i in range(len(g_optim_dict['parts_optim'])):
            #     ckpt[f'g_optim_part_{i}'] = g_optim_dict['parts_optim'][i].state_dict()

            torch.save(
                ckpt,
                os.path.join(run_dir, "checkpoints", f"step{str(step).zfill(5)}.pt"),
            )
