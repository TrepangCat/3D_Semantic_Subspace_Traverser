import argparse
import torch
import os
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np


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
        help="checkpoint of GAN",
    )
    parser.add_argument(
        "--ae_ckpt_path",
        type=str,
        default='./checkpoint/vae_ckpt-0186/step00000200.pt',
        help="path to the VAE ckpt",
    )
    parser.add_argument(
        "--num_generated",
        type=int,
        default=10,
        help="num of generated results",
    )
    tmp_args = parser.parse_args()

    ckpt_path = tmp_args.ckpt
    ae_ckpt_path = tmp_args.ae_ckpt_path
    num_generated = tmp_args.num_generated

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    device = args.device

    seeds = list(range(num_generated))
    g_type = "g"  # 'g' or 'g_ema'

    # Import!!!
    from train_GAN import Generator, ShapeAE64, save_mesh_from_points_for_pool

    # Network
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
    ae_ckpt = torch.load(ae_ckpt_path)
    AE.load_state_dict(ae_ckpt["ae"])

    outdir = os.path.join(os.path.dirname(ckpt_path),
                          f'generate_{g_type}_r={resolution_coord}_threshold={threshold}_'
                          + str(ckpt["step"]).zfill(5)) + f'_num{len(seeds)}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(f"result path: {outdir}")

    with torch.no_grad() and Pool(4) as p:

        def generation_list(seeds, save_dir):
            for seed_idx, seed in enumerate(seeds):
                torch.manual_seed(seed)  # make sure the same random vecter being generated
                shape_code = G.sample(1)

                # dir_name = os.path.basename(save_dir)
                # io.savemat(os.path.join(save_dir, f'sdfmesh_{dir_name}-seed{seed}-index{seed_idx}.mat'),
                #            {'shape_code': shape_code.cpu().numpy()})

                pred_occ_list = []
                for points in grid_points_split:
                    pred_occ = AE.decoder(shape_code, points).squeeze(0).detach().cpu()
                    pred_occ_list.append(pred_occ)
                all_pred_occ = torch.cat(pred_occ_list, dim=0)

                input_dict = {'coord_occ': all_pred_occ,
                              'save_dir': save_dir,
                              'seed': seed,
                              'index': seed_idx,
                              'no_png': False,
                              'resolution': resolution_coord,
                              'threshold': threshold}

                yield input_dict


        res_list = p.imap_unordered(save_mesh_from_points_for_pool, generation_list(seeds, outdir))

        bar = tqdm(total=len(seeds), desc='Visualization:')
        for res in res_list:
            bar.update()

    exit(0)
