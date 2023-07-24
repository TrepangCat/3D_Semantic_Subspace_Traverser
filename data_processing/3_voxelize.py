import trimesh
import numpy as np
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import traceback
import voxels
import argparse
from tqdm import tqdm
import gc


def voxelize(in_path, res):
    try:

        filename = os.path.join(in_path, 'voxelization_{}.npy'.format(res))

        if os.path.exists(filename):
            return

        mesh = trimesh.load(in_path + '/isosurf_scaled.off', process=False)
        occupancies = voxels.VoxelGrid.from_mesh(mesh, res, loc=[0, 0, 0], scale=1).data
        occupancies = np.reshape(occupancies, -1)

        # gc.collect()

        if not occupancies.any():
            raise ValueError('No empty voxel grids allowed.')

        occupancies = np.packbits(occupancies)
        np.save(filename, occupancies)

    except Exception as err:
        path = os.path.normpath(in_path)
        print('Error with {}: {}'.format(path, traceback.format_exc()))
    # print('finished {}'.format(in_path))


def get_mission_list(list):
    l = len(list)
    for x in tqdm(sorted(list), total=l):
        yield x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run voxalization'
    )
    parser.add_argument('-res', default=32, type=int)
    args = parser.parse_args()

    path_list = [
        # '/home/brl/dataDisk2/wrw/dataset/ShapeNetDataSet_subset/ShapeNetCore.v1_processed/02691156/train',
        # '/home/brl/dataDisk2/wrw/dataset/ShapeNetDataSet_subset/ShapeNetCore.v1_processed/02958343/train/origin',
        # '/home/brl/dataDisk2/wrw/dataset/ShapeNetDataSet_subset/ShapeNetCore.v1_processed/02958343/train/extend',
        '/home/brl/dataDisk2/wrw/dataset/ShapeNetDataSet_subset/ShapeNetCore.v1_processed_hsp',
    ]

    # with Pool(int(mp.cpu_count() * 0.5)) as p:
    with Pool(8) as p:
        for path in path_list:
            off_list = glob.glob(os.path.join(path, '**', '*isosurf_scaled.off'), recursive=True)
            off_list = [os.path.dirname(x) for x in off_list]

            res_list = p.imap_unordered(partial(voxelize, res=args.res), get_mission_list(off_list), chunksize=1)
            # res_list = p.imap(partial(voxelize, res=args.res), get_mission_list(off_list), chunksize=1)
            # res_list = p.map(partial(voxelize, res=args.res), get_mission_list(off_list), chunksize=1)

            # 下面是为imap版本设计的
            bar = tqdm(range(len(off_list)), position=0)
            for res in res_list:
                bar.update()