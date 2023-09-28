import argparse
import glob
import os
import re
from multiprocessing import Pool

import mcubes  # 这个应该是来自 pymcubes 包
import numpy as np
import pyrender
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
import sys

os.environ['PYOPENGL_PLATFORM'] = 'egl'


class Dataset:
    def __init__(self, root, num_samples=20000, res=128, alpha=0.1):
        self.root = root
        self.num_samples = num_samples
        self.res = res

        npz_path = f'**/*_{alpha}_samples100000.npz'

        # if 'train' in self.root:
        #     npz_path = f'**/*_{alpha}_samples100000.npz'
        # else:
        #     npz_path = f'**/train/**/*_{alpha}_samples100000.npz'
        self.data_list = glob.glob(os.path.join(self.root, npz_path), recursive=True)
        self.data_list = sorted(self.data_list)
        # self.data_list = self.data_list[:100]

    def __getitem__(self, index):
        try:
            npz_name = self.data_list[index]

            # load vox
            dir_name = os.path.dirname(npz_name)

            vox = np.load((os.path.join(dir_name, f'voxelization_{self.res}.npy')), allow_pickle=True)

            vox = np.unpackbits(vox)
            size = int(np.cbrt(vox.shape))
            vox = np.reshape(vox, (size,) * 3)
            vox = vox[np.newaxis, :]  # 增加“通道”这个维度

            # cat_name = os.path.basename(dir_name)
            # lack_armrest = False
            # lack_legver = False
            # lack_leghor = False
            # if cat_name.__contains__('armrest'):
            #     lack_armrest = True
            # if cat_name.__contains__('legver'):
            #     lack_legver = True
            # if cat_name.__contains__('leghor'):
            #     lack_leghor = True

            # load boundary_samples
            # points = []
            coords = []
            occupancies = []

            boundary_samples_npz = np.load(npz_name)
            # boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_occupancies = boundary_samples_npz['occupancies']
        except Exception:
            print(npz_name)
            print('Error!')
            exit(0)

        # change the orders
        subsample_indices = np.random.randint(0, len(boundary_sample_coords), self.num_samples)
        # points.extend(boundary_sample_points[subsample_indices])
        coords.extend(boundary_sample_coords[subsample_indices])
        occupancies.extend(boundary_sample_occupancies[subsample_indices])

        data = {'grid_coords': np.array(coords, dtype=np.float32),
                'occupancies': np.array(occupancies, dtype=np.float32),
                # 'points': np.array(points, dtype=np.float32),
                'vox': np.array(vox, dtype=np.float32),
                'path': dir_name,
                'index': index,
                # 'lack_armrest': lack_armrest,
                # 'lack_legver': lack_legver,
                # 'lack_leghor': lack_leghor,
                }

        return data

    def __len__(self):
        return len(self.data_list)


class Encoder64(torch.nn.Module):
    def __init__(self,
                 gen_channels=128):
        super().__init__()

        self.gen_channels = gen_channels

        # accepts 128**3 res input
        self.layer_1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),  # out: 64
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, 3, padding=1),  # out: 64
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),  # out: 32
        )
        self.layer_2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1),  # out: 32
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, 3, padding=1),  # out: 32
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),  # out: 16
        )
        self.layer_3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),  # out: 16
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, 3, padding=1),  # out: 16
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            # nn.MaxPool3d(2),  # out: 16
        )
        self.layer_4 = nn.Sequential(
            nn.Conv3d(64, gen_channels * 2, 3, padding=1),  # out: 16
            # nn.Tanh()
        )

        # # accepts 128**3 res input
        # self.layer_1 = nn.Sequential(
        #     nn.Conv3d(1, 16, 3, padding=1),  # out: 128
        #     nn.BatchNorm3d(16),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(16, 16, 3, padding=1),  # out: 128
        #     nn.BatchNorm3d(16),
        #     nn.LeakyReLU(),
        #     nn.MaxPool3d(2),  # out: 64
        # )
        # self.layer_2 = nn.Sequential(
        #     nn.Conv3d(16, 32, 3, padding=1),  # out: 64
        #     nn.BatchNorm3d(32),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(32, 32, 3, padding=1),  # out: 64
        #     nn.BatchNorm3d(32),
        #     nn.LeakyReLU(),
        #     nn.MaxPool3d(2),  # out: 32
        # )
        # self.layer_3 = nn.Sequential(
        #     nn.Conv3d(32, 64, 3, padding=1),  # out: 32
        #     nn.BatchNorm3d(64),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(64, 64, 3, padding=1),  # out: 32
        #     nn.BatchNorm3d(64),
        #     nn.LeakyReLU(),
        #     nn.MaxPool3d(2),  # out: 16
        # )
        # self.layer_4 = nn.Sequential(
        #     nn.Conv3d(64, gen_channels * 2, 3, padding=1),  # out: 16
        #     # nn.Tanh()
        # )

    def forward(self, x):
        # p: sample points
        # x: input voxel
        if x.ndim == 4:
            x = x.unsqueeze(1)  # [n, r, r, r] => [n, 1, r, r, r]

        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)

        return out


class Decoder64(torch.nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 gen_channels=128):
        super().__init__()

        # feature_size = gen_channels * 1
        feature_size = gen_channels * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.bn_0 = nn.BatchNorm1d(hidden_dim)

        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)

        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.bn_2 = nn.BatchNorm1d(hidden_dim)

        # self.fc_3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        # self.bn_3 = nn.BatchNorm1d(hidden_dim)
        #
        # self.fc_4 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        # self.bn_4 = nn.BatchNorm1d(hidden_dim)

        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.LeakyReLU()

        # displacment = 0.0722  # 为什么 IF-NET 中设置的是这个值？ 这个值对应128分辨率
        displacment = 0.0536  # (0.0722 + 0.035) / 2 = 0.0535

        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, shape_code, p):
        # p_features = p.transpose(1, -1)
        xyz = p.permute(0, 2, 1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)

        # shape_code.shape=(B, c, r, r, r)
        features = F.grid_sample(shape_code, p, align_corners=True)  # align_corners=True 使得函数与 pytorch 1.1.0 版本的表现一致。
        # features.shape=(B, c, 1, 7, num_samples)

        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        # features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)
        features = torch.cat([features, xyz], dim=1)

        out = self.fc_0(features)
        out = self.bn_0(out)
        out = self.actvn(out)

        out = self.fc_1(out)
        out = self.bn_1(out)
        out = self.actvn(out)

        out = self.fc_2(out)
        out = self.bn_2(out)
        out = self.actvn(out)

        # out = self.fc_3(out)
        # out = self.bn_3(out)
        # out = self.actvn(out)
        #
        # out = self.fc_4(out)
        # out = self.bn_4(out)
        # out = self.actvn(out)

        out = self.fc_out(out)
        out = out.squeeze(1)  # 输出的是采样点是否在形状内部的概率

        return out


class ShapeAE64(torch.nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 gen_channels=128):
        super().__init__()

        self.gen_channels = gen_channels

        self.encoder = Encoder64(gen_channels=gen_channels)
        self.decoder = Decoder64(hidden_dim=hidden_dim, gen_channels=gen_channels)

    def re_parameter(self, meanstd_code):
        m, s = meanstd_code[:, :self.gen_channels], meanstd_code[:, self.gen_channels:]
        dist = torch.distributions.Normal(m, s.exp())  # 因为标准差是非负数，所以我们把s 看作 Log(std)。通过s.exp()将其变为真正的标准差
        z = dist.rsample()

        return z, dist

    def forward(self, x, p):
        meanstd_code = self.encoder(x)
        z, dist = self.re_parameter(meanstd_code)
        predict_occ = self.decoder(z, p)

        return predict_occ, dist, meanstd_code


# --------------------------------------------------------


def save_mesh_from_points_for_pool(input):
    coord_occ = input['coord_occ']
    save_dir = input['save_dir']
    index = input['index']
    no_png = input['no_png']
    resolution = input['resolution']
    threshold = input['threshold']

    save_mesh_from_points(coord_occ=coord_occ,
                          save_dir=save_dir,
                          index=index,
                          no_png=no_png,
                          resolution=resolution,
                          threshold=threshold,
                          )


def save_mesh_from_points(coord_occ, save_dir, index, no_png=False, resolution=256, threshold=0.5):
    assert coord_occ.ndim == 1
    assert isinstance(coord_occ, torch.Tensor) or isinstance(coord_occ, np.ndarray)

    if isinstance(coord_occ, torch.Tensor):
        coord_occ = coord_occ.numpy()

    dir = save_dir
    if not os.path.exists(dir):
        os.makedirs(dir)

    dir_name = os.path.basename(save_dir)
    name_obj = os.path.join(dir, f'sdfmesh_{dir_name}-{index}.obj')

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
    logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default='./sample_data/ShapeNetCore.v1_processed_tmp',
        help="path to the dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20000,
        help="num of sample for dataset",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="alpha value for dataset",
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
        default=32,
        help="grid size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="train steps",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=10,
        help="sample log period",
    )
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=10,
        help="checkpoint save period",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        # default=1e-4,
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
        "--code_channels",
        type=int,
        default=8,
        help="num of shape code channels for generator",
    )
    parser.add_argument(
        "--no_png",
        type=str2bool,
        default=False,
        help="whether to train GAN",
    )
    parser.add_argument(
        "--kld_loss_weight",
        type=float,
        default=10.0,
        help="weight of kld_loss",
    )

    args = parser.parse_args()
    print(args)

    device = args.device

    # Set the seed. Make the experiments repeatable.
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True

    # models
    AE = ShapeAE64(
        gen_channels=args.code_channels,
    ).to(device).train()

    # optimizers
    ae_optim = torch.optim.Adam(
        AE.parameters(),
        lr=args.lr,
        betas=(0.5, 0.99),
    )

    # load checkpoint
    start_step = 0
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        start_step = ckpt["step"]
        AE.load_state_dict(ckpt["ae"])
        ae_optim.load_state_dict(ckpt['ae_optim'])

    # data
    dataset = Dataset(args.path, args.num_samples, res=64, alpha=args.alpha)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        # drop_last=False,
        num_workers=args.n_workers,
        prefetch_factor=16,
    )

    # Define the point_coords
    min_coord = -0.5
    max_coord = 0.5
    resolution_coord = 128
    grid_points = create_grid_points_from_bounds(min_coord, max_coord, resolution_coord)
    grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()  # 为什么要写这一行？

    a = max_coord + min_coord
    b = max_coord - min_coord

    grid_coords = 2 * grid_points - a  # 将值域从 [-0.5, 0.5] 变换到 [-1.0, 1.0]
    grid_coords = grid_coords / b

    grid_coords = torch.from_numpy(grid_coords).to(device, dtype=torch.float)
    grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(device)
    grid_points_split = torch.split(grid_coords, 500000, dim=1)

    # train utils_sdfstylegan
    # ema = partial(accumulate, decay=0.5 ** (args.batch / (10 * 1000)))
    # if args.fid:  # 实际上因为下面这个是二维的网络，所以不能用。
    #     compute_fid = fid.get_fid_fn(dataset, device=device)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(args.output_dir):
        prev_run_dirs = [x for x in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    is_resume = '-resumed' if args.ckpt is not None else ''
    run_dir = os.path.join(args.output_dir, f'{cur_run_id:04d}-AE-batch{args.batch}-steps{args.steps}' + is_resume)
    assert not os.path.exists(run_dir)
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    tb_writer = SummaryWriter(run_dir)
    print(f"training log directory: {run_dir}")

    first_visualization = True

    # print module summary
    with open(os.path.join(run_dir, "setting.txt"), 'w') as f:
        f.write(f'{sys.argv[0]}\n\n')

        example = next(iter(loader))
        example_p = example.get('grid_coords').to(device)  # 散点的坐标
        example_vox = example.get('vox').to(device)  # 体素
        # _, str_ae = print_module_summary(module=AE, inputs=(example_vox, example_p))
        # f.write(str_ae)

    # train loop
    # iterator = tqdm(range(args.steps), initial=start_step, ncols=120)
    ae_loss_avg = 0
    ae_loss = torch.Tensor([0]).to(device)
    kld_loss_avg = 0
    kld_loss = torch.Tensor([0]).to(device)
    total_loss_avg = 0
    total_loss = torch.Tensor([0]).to(device)

    epoch_iter = tqdm(range(args.steps),
                      initial=start_step,
                      # ncols=80,
                      # ncols=100,
                      # position=1,
                      # position=0,
                      )

    for step in epoch_iter:
        # for step in (iterator := tqdm(range(args.steps), initial=start_step)):

        step = step + start_step + 1
        # step = step + start_step
        if step > args.steps:
            break

        iterator = tqdm(loader,
                        # ncols=80,
                        # ncols=100,
                        # position=0,
                        )

        for real_data in iterator:
            p = real_data.get('grid_coords').to(device)  # 散点的坐标
            occ = real_data.get('occupancies').to(device)  # 散点的内外情况
            vox = real_data.get('vox').to(device)  # 体素

            # AE
            AE.train().requires_grad_(True)

            ae_logits, dist, meadstd_code = AE(vox, p)

            # bce_loss
            loss_i = F.binary_cross_entropy_with_logits(ae_logits, occ, reduction='none')
            ae_loss = loss_i.sum(-1).mean()
            total_loss = ae_loss

            # kld_loss
            p = torch.distributions.Normal(torch.zeros_like(dist.mean), torch.ones_like(dist.scale))
            kld_loss = torch.distributions.kl.kl_divergence(dist, p).mean() * args.kld_loss_weight
            total_loss = total_loss + kld_loss

            ae_optim.zero_grad()
            total_loss.backward()
            ae_optim.step()

            # log
            iterator.set_description(
                f"total:{total_loss.item():.4f}, bce:{ae_loss.item():.4f}, kld:{kld_loss.item():.4f}"
            )

            len_loader = len(loader)
            total_loss_avg += total_loss.item() / len_loader
            ae_loss_avg += ae_loss.item() / len_loader
            kld_loss_avg += kld_loss.item() / len_loader

        tb_writer.add_scalar("loss_AE/total", total_loss_avg, step)
        tb_writer.add_scalar("loss_AE/bce", ae_loss_avg, step)
        tb_writer.add_scalar("loss_AE/kld", kld_loss_avg, step)

        total_loss_avg = 0
        ae_loss_avg = 0
        kld_loss_avg = 0

        done = args.steps == step
        if step % args.sample_every == 1 or done or args.sample_every == 1:
            with torch.no_grad():
                AE.eval().requires_grad_(False)

                # visualize the VAE results
                ae_visual_list = [
                    torch.Tensor(dataset[0]['vox']).unsqueeze(0).to(device),
                    torch.Tensor(dataset[1]['vox']).unsqueeze(0).to(device),
                    torch.Tensor(dataset[2]['vox']).unsqueeze(0).to(device),
                    torch.Tensor(dataset[3]['vox']).unsqueeze(0).to(device),
                    torch.Tensor(dataset[4]['vox']).unsqueeze(0).to(device),
                    # torch.Tensor(dataset[0]['vox']).unsqueeze(0).to(device),
                    # torch.Tensor(dataset[1]['vox']).unsqueeze(0).to(device),
                    # torch.Tensor(dataset[2]['vox']).unsqueeze(0).to(device),
                    # torch.Tensor(dataset[3]['vox']).unsqueeze(0).to(device),
                    # torch.Tensor(dataset[4]['vox']).unsqueeze(0).to(device),
                ]

                for i in range(min(5, len(vox))):
                    ae_visual_list += [vox.split(1)[i]]  # 取出一部分数据。（这部分是随机的）

                mission_list = []

                index = 0
                for vox in ae_visual_list:
                    input_ae = vox.to(device)

                    pred_occ_list = []
                    for points in grid_points_split:
                        pred_occ = AE(input_ae, points)[0].squeeze(0).detach().cpu()
                        pred_occ_list.append(pred_occ)
                    all_pred_occ = torch.cat(pred_occ_list, dim=0)

                    save_dir = os.path.join(run_dir, f'fakes_AE_{step:08d}')

                    index += 1

                    input_dict = {'coord_occ': all_pred_occ,
                                  'save_dir': save_dir,
                                  'index': index,
                                  'no_png': args.no_png,
                                  'resolution': resolution_coord,
                                  'threshold': 0.0,
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
                    bar = tqdm(len(mission_list), desc='AE_visualization')
                    for res in res_list:
                        bar.update()

        if step % args.ckpt_every == 1 or done or args.ckpt_every == 1:
            if not os.path.exists(os.path.join(run_dir, "checkpoints")):
                os.makedirs(os.path.join(run_dir, "checkpoints"))

            ckpt = {
                "step": step,
                "args": args,
                "ae": AE.eval().requires_grad_(False).state_dict(),
                "ae_optim": ae_optim.state_dict(),
            }

            torch.save(
                ckpt,
                os.path.join(run_dir, "checkpoints", f"step{str(step).zfill(8)}.pt"),
            )
