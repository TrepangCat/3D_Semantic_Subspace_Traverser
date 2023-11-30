import argparse
import os
import scipy.io as io
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


# Define the point_coords
min_coord = -0.5
max_coord = 0.5
resolution_coord = 64
grid_points = create_grid_points_from_bounds(min_coord, max_coord, resolution_coord)
grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()  # 为什么要写这一行？

a = max_coord + min_coord
b = max_coord - min_coord

grid_coords = 2 * grid_points - a  # 将值域从 [-0.5, 0.5] 变换到 [-1.0, 1.0]
grid_coords = grid_coords / b

grid_coords = torch.from_numpy(grid_coords).to('cuda', dtype=torch.float)
grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to('cuda')
grid_points_split = torch.split(grid_coords, 500000, dim=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ckpt",
        type=str,
        help="checkpoint file to continue training",
    )
    parser.add_argument(
        "--path",
        type=str,
        default='./sample_data/ShapeNetCore.v1_processed_tmp',
        help="path to the dataset",
    )
    tmp_args = parser.parse_args()
    ckpt_path = tmp_args.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]

    data_path = tmp_args.path
    output_dir = data_path + '_encoded'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = args.device

    # import!!!
    from train_VAE import Dataset, ShapeAE64 as ShapeAE, save_mesh_from_points_for_pool

    # Dataset
    batch_size = 4
    dataset = Dataset(data_path, args.num_samples, res=64, alpha=args.alpha)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.n_workers,
        prefetch_factor=8,
    )
    assert len(loader) * batch_size >= len(dataset)

    # Model
    AE = ShapeAE(
        gen_channels=args.code_channels,
    ).to(device).eval().requires_grad_(False)
    AE.load_state_dict(ckpt["ae"])

    for data_batch in tqdm(loader):
        p = data_batch.get('grid_coords').to(device)  # 散点的坐标
        occ = data_batch.get('occupancies').to(device)  # 散点的内外情况
        vox = data_batch.get('vox').to(device)  # 体素
        data_name_list = [os.path.basename(name) for name in data_batch.get('path')]

        meanstd_code = AE.encoder(vox)
        shape_code = meanstd_code[:, :AE.gen_channels]

        for i in range(len(vox)):
            # torch.save(shape_code[i], os.path.join(output_dir, data_name_list[i] + '.ae'))

            io.savemat(os.path.join(output_dir, data_name_list[i] + '.mat'),
                       {'shape_code': shape_code[i].cpu().numpy()})

            # # visualization for check
            # tmp_code = shape_code[i].unsqueeze(0)
            # pred_occ_list = []
            # for points in grid_points_split:
            #     pred_occ = AE.decoder(tmp_code, points).squeeze(0).detach().cpu()
            #     pred_occ_list.append(pred_occ)
            # all_pred_occ = torch.cat(pred_occ_list, dim=0)
            #
            # input_dict = {'coord_occ': all_pred_occ,
            #               'save_dir': output_dir,
            #               'index': data_name_list[i],
            #               'no_png': False,
            #               'resolution': resolution_coord,
            #               'threshold': 0.0
            #               }
            #
            # save_mesh_from_points_for_pool(input_dict)

    print('finished')
    exit(0)
