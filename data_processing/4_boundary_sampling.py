import argparse
import glob
import os
import traceback
from multiprocessing import Pool, cpu_count

import numpy as np
import trimesh
from tqdm import tqdm

import implicit_waterproofing as iw


def equal(x, y):
    try:
        assert abs(x - y) <= 1e-8
        return True
    except Exception:
        raise ValueError(f'{x}!={y}')


def boundary_sampling(path):
    try:
        out_file = path + f'/boundary_{args.sigma}_samples{sample_num}.npz'

        if os.path.exists(out_file):
            # if os.path.exists(path + f'/boundary_{args.sigma}_samples{sample_num}.npz'):
            return

        off_path = path + '/isosurf_scaled.off'

        if os.path.exists(out_file):
            return

        if not os.path.exists(off_path):
            raise FileNotFoundError(f'{off_path} not found.')

        mesh = trimesh.load(off_path)

        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()  # 求取 x、y、z 三个方向上的最长边
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2  # 求取整个长方体空间的中心

        mesh.apply_translation(-centers)  # 将中心移动到零点 [0,0,0]
        mesh.apply_scale(1 / total_size)  # 将长方体空间的最大边长变为 1

        # 验证归一化是否正确
        new_centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
        for i in range(3):
            assert equal(mesh.bounds[0][i] + mesh.bounds[1][i], 0.0)
            assert equal(new_centers[i], 0.0)
        assert equal(abs((mesh.bounds[1] - mesh.bounds[0]).max()), 1.0)

        points = mesh.sample(sample_num)

        boundary_points = points + args.sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]  # 为什么这里要换过来？

        grid_coords = 2 * grid_coords  # 这个地方把采样点，扩了两倍？因为mesh的最大最小值在[-0.5, 0.5]所以采样点加了个倍。

        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        # np.savez(out_file, points=boundary_points, occupancies=occupancies, grid_coords=grid_coords)
        # np.savez(out_file, points=None, occupancies=occupancies, grid_coords=grid_coords)
        np.savez(out_file, occupancies=occupancies, grid_coords=grid_coords)
        # print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


def get_mission_list(list):
    for x in list:
        yield x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run boundary sampling')
    parser.add_argument('-sigma', default=0.1, type=float)
    parser.add_argument('--processed_data_dir', type=str, default='../sample_data/ShapeNetCore.v1_processed_tmp')
    args = parser.parse_args()

    sample_num = 100000
    # sample_num = 200000

    ROOT = args.processed_data_dir

    with Pool(4) as p:
        # with Pool(int(cpu_count() * 0.5)) as p:
        obj_file_list = [os.path.dirname(path) for path in
                         glob.glob(os.path.join(ROOT, '**/isosurf_scaled.off'), recursive=True)]

        res_list = p.imap_unordered(boundary_sampling, get_mission_list(obj_file_list), chunksize=2)

        bar = tqdm(range(len(obj_file_list)))
        for res in res_list:
            bar.update()
