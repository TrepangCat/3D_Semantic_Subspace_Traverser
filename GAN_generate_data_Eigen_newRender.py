import argparse
import os

import numpy as np
import torch
from skimage import io as im_io
from tqdm import tqdm

import utils_3D_fromsefa
from my_rendering.render_api import render_trimesh_v1
import trimesh
from trimesh.exchange.obj import export_obj


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


def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


# ============================
# Define the point_coords
min_coord = -0.5
max_coord = 0.5
resolution_coord = 128
grid_points = create_grid_points_from_bounds(min_coord, max_coord, resolution_coord)
grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()  # 为什么要写这一行？——交换x轴和z轴的位置，使得可视化正常。

a = max_coord + min_coord
b = max_coord - min_coord

grid_coords = 2 * grid_points - a  # 将值域从 [-0.5, 0.5] 变换到 [-1.0, 1.0]
grid_coords = grid_coords / b

grid_coords = torch.from_numpy(grid_coords).to('cuda', dtype=torch.float)
grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to('cuda')
grid_points_split = torch.split(grid_coords, 500000, dim=1)
# ============================

threshold = 0.0  # threshold of the isosurface

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ckpt",
        type=str,
        help="checkpoint file to continue training",
    )
    parser.add_argument(
        "--ae_ckpt_path",
        type=str,
        default='./checkpoint/vae_ckpt-0186/step00000200.pt',
        help="path to the VAE ckpt",
    )
    parser.add_argument(
        "--traverse_range",
        type=float,
        default=6.0,
        help="the traverse range",
    )
    parser.add_argument(
        "--intermediate_points",
        type=int,
        default=7,
        help="the number of intermediate points during traversing",
    )
    parser.add_argument('--tvs_seed', nargs='+', type=int, default=1,
                        help="which seed to generate shapes for manipulation")
    tmp_args = parser.parse_args()

    ckpt_path = tmp_args.ckpt
    ae_ckpt_path = tmp_args.ae_ckpt_path
    traverse_seed = tmp_args.tvs_seed
    traverse_range = tmp_args.traverse_range
    intermediate_points = tmp_args.intermediate_points
    truncation = 1.0

    traverse_seed.sort()

    print(f'traversing_seed={traverse_seed}')
    print(f'traversing_range={traverse_range}')
    print(f'intermediate_points={intermediate_points}')
    print(f'truncation={truncation}')

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    device = args.device

    # seeds = list(range(6780))  # 这里的种子也可以考虑手动设置
    # seeds = list(range(3000))  # 这里的种子也可以考虑手动设置，车辆数据的test集的数目=1500，3000是其两倍。
    # seeds = list(range(2712))  # 这里的种子也可以考虑手动设置  为椅子设计的数目
    # seeds = list(range(200))  # 这里的种子也可以考虑手动设置
    # seeds = [293,301,314,320,339]

    g_type = "g"  # 'g' or 'g_ema'

    # Import!!!
    from train_GAN import Generator, ShapeAE64, mesh_from_logits

    # Network
    # from train_AE_0060 import ShapeAE64
    G = Generator(
        # size=args.size,
        n_basis=args.n_basis,
        noise_dim=args.noise_dim,
        # base_channels=args.base_channels,
        code_channels=args.code_channels,
        max_channels=args.max_channels,
        device=device,
    ).to(device).eval().requires_grad_(False)
    G.load_state_dict(ckpt[g_type])

    AE = ShapeAE64(
        gen_channels=args.code_channels,
    ).to(device).eval().requires_grad_(False)
    # ae_ckpt_path = '/home/brl/data_disk_4t/code/stylegan_series/A_new_gan_project_july/run_CubeCodeCraft/0114-AE-batch16-steps200/checkpoints/step00000200.pt'
    # ae_ckpt_path = '/home/brl/data_disk_4t/code/stylegan_series/A_new_gan_project_july/run_CubeCodeCraft/0105-AE-batch16-steps200/checkpoints/step00000200.pt'
    # ae_ckpt_path = '/home/brl/data_disk_4t/code/stylegan_series/A_new_gan_project_july/run_CubeCodeCraft/0097-AE-batch32-steps200/checkpoints/step00000200.pt'
    ae_ckpt = torch.load(ae_ckpt_path)
    AE.load_state_dict(ae_ckpt["ae"])

    outdir = os.path.join(os.path.dirname(ckpt_path),
                          f'generateEigen_{g_type}_r={resolution_coord}_threshold={threshold}_' + str(
                              ckpt["step"]).zfill(5))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(f"result path: {outdir}")

    # 下面是编辑代码
    # traverse_seed = list(range(100))  # 这里的种子也可以考虑手动设置
    # traverse_seed = [723, ]
    # traverse_seed = [0, 36, 494, 544, 191, 286, 634, 366, 392, 605, 603, 781, 652, 320, 884, 965, 610, 645, 1541, 1553, 1564]
    # traverse_seed = [7, 9, 12, 47, 88,]  # 这里的种子也可以考虑手动设置
    # traverse_seed = [14,23,29,33,55]  # 这里的种子也可以考虑手动设置
    # traverse_seed = [24,45,48]  # 这里的种子也可以考虑手动设置
    # traverse_seed = [2,8,16]  # 这里的种子也可以考虑手动设置
    # traverse_seed = [
    #     293,301,314,320,339
    # ]  # 这里的种子也可以考虑手动设置

    traverse_samples = len(traverse_seed)
    offsets = np.linspace(-traverse_range, traverse_range, intermediate_points)

    # create a html file to show the editing results. 创建 html 文件，用于展示图片。
    es, zs = G.sample_latents(1)
    batch, n_layers, n_dim = zs.shape
    vizer1 = utils_3D_fromsefa.HtmlPageVisualizer(num_rows=len(traverse_seed) * (n_layers * n_dim + 1),
                                                  num_cols=intermediate_points + 1,
                                                  viz_size=256)
    vizer2 = utils_3D_fromsefa.HtmlPageVisualizer(num_rows=n_layers * n_dim * (len(traverse_seed) + 1),
                                                  num_cols=intermediate_points + 1,
                                                  viz_size=256)
    headers = [''] + [f'Distance {d:.2f}' for d in offsets]
    vizer1.set_headers(headers)
    vizer2.set_headers(headers)

    iterator = tqdm(range(len(traverse_seed) * n_layers * n_dim * len(offsets)), initial=0)
    for seed_idx, seed in enumerate(traverse_seed):
        vizer1.set_cell(row_idx=(n_layers * n_dim + 1) * seed_idx,
                        col_idx=0,
                        text=f'seed={seed:04d}',
                        highlight=True)

        torch.manual_seed(seed)  # make sure the same random vecter being generated
        es, zs = G.sample_latents(1, truncation=truncation)
        # es, zs = g_ema.sample_latents(1, lack_armrest=False, lack_legver=False, lack_leghor=False,
        #                               truncation=truncation)
        # es, zs = g_ema.sample_latents(1, truncation=truncation)
        batch, n_layers, n_dim = zs.shape

        for i_layer in range(n_layers):
            # if i_layer > 0:  # 有必要的情况下，可以用这个代码跳过某些layer
            #     continue

            for i_dim in range(n_dim):
                # print(f" seed{seed} - layer {i_layer} - dim {i_dim}")
                vizer1.set_cell(row_idx=(n_layers * n_dim + 1) * seed_idx + 1 + i_layer * n_dim + i_dim,
                                col_idx=0,
                                text=f'layer {i_layer:03d} - dim {i_dim:03d}')

                vizer2.set_cell(
                    row_idx=i_layer * n_dim * (len(traverse_seed) + 1) + i_dim * (len(traverse_seed) + 1),
                    col_idx=0,
                    text=f'layer {i_layer:03d} - dim {i_dim:03d}',
                    highlight=True)

                vizer2.set_cell(
                    row_idx=i_layer * n_dim * (len(traverse_seed) + 1) + i_dim * (len(traverse_seed) + 1)
                            + 1 + seed_idx,
                    col_idx=0,
                    text=f'seed={seed:04d}')

                shape_codes = []
                for offset_idx, offset in enumerate(offsets):
                    img_path = os.path.join(os.path.join(outdir, 'eigen_images',
                                                         f'seed={seed:04d} - layer={i_layer:03d} - dim={i_dim:03d} - z={offset}.png'))

                    if not os.path.exists(os.path.dirname(img_path)):
                        os.makedirs(os.path.dirname(img_path))

                    if not os.path.exists(img_path):
                        _zs = zs.clone()
                        _zs[:, i_layer, i_dim] = offset
                        with torch.no_grad():
                            shape_code = G(es, _zs).to(device)

                            pred_occ_list = []
                            for points in grid_points_split:
                                pred_occ = AE.decoder(shape_code, points).squeeze(0).detach().cpu()
                                pred_occ_list.append(pred_occ)
                            all_pred_occ = torch.cat(pred_occ_list, dim=0)

                            mesh = mesh_from_logits(all_pred_occ, resolution=resolution_coord, t=0.0)
                            if len(mesh.vertices) == 0:
                                continue

                            _write_mesh_as_obj(mesh, output_path=img_path[:-4] + '.obj')

                            image = render_trimesh_v1(mesh)
                            im_io.imsave(fname=img_path, arr=image)

                            # image = from_trimesh_to_img(mesh, img_path, rotate_x=20, rotate_y=-120, rewrite=False)
                            # io.imsave(os.path.join(os.path.join(outdir, 'semantic', f'seed={seed:04d} - layer={i_layer:03d} - dim={i_dim:03d} - z={offset}.png')))

                    image = im_io.imread(img_path)

                    vizer1.set_cell(
                        row_idx=(n_layers * n_dim + 1) * seed_idx + 1 + i_layer * n_dim + i_dim,
                        col_idx=offset_idx + 1,
                        image=image)

                    vizer2.set_cell(
                        row_idx=i_layer * n_dim * (len(traverse_seed) + 1) + i_dim * (len(traverse_seed) + 1)
                                + 1 + seed_idx,
                        col_idx=offset_idx + 1,
                        image=image)

                    iterator.set_description(f" seed{seed} - layer {i_layer} - dim {i_dim} - offset {offset}")
                    iterator.update()

    vizer1.save(
        os.path.join(outdir, f'SeedFirst-Nseeds-{len(traverse_seed)}_' +
                     f'L{n_layers}_D{i_dim}_distance{traverse_range}_inter{intermediate_points}_seeds{traverse_seed[0]}.html'))
    vizer2.save(os.path.join(outdir,
                             f'SemanticFirst-Nseeds-{len(traverse_seed)}_' +
                             f'L{n_layers}_D{i_dim}_distance{traverse_range}_inter{intermediate_points}_seeds{traverse_seed[0]}.html'))

    # vizer1.save(
    #     os.path.join(outdir, f'SeedFirst-Nseeds-{len(traverse_seed)}_' +
    #                  f'L{n_layers}_D{i_dim}_distance{traverse_range}_inter{intermediate_points}_seeds{traverse_seed[0]}_' +
    #                  f'{traverse_seed[1]}_{traverse_seed[2]}_{traverse_seed[3]}_{traverse_seed[4]}.html'))
    # vizer2.save(
    #     os.path.join(outdir,
    #                  f'SemanticFirst-Nseeds-{len(traverse_seed)}_' +
    #                  f'L{n_layers}_D{i_dim}_distance{traverse_range}_inter{intermediate_points}_seeds{traverse_seed[0]}_' +
    #                  f'{traverse_seed[1]}_{traverse_seed[2]}_{traverse_seed[3]}_{traverse_seed[4]}.html'))

    exit(0)
