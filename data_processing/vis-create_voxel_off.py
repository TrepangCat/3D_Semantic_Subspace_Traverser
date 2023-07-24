import argparse
import os
import numpy as np

from voxels import VoxelGrid


def create_voxel_off(path):
    voxel_path = path + '/voxelization_{}.npy'.format(res)
    off_path = path + '/voxelization_{}.off'.format(res)

    if unpackbits:
        occ = np.unpackbits(np.load(voxel_path))
        voxels = np.reshape(occ, (res,) * 3)
    else:
        voxels = np.reshape(np.load(voxel_path)['occupancies'], (res,) * 3)

    loc = ((min + max) / 2,) * 3
    scale = max - min

    VoxelGrid(voxels, loc, scale).to_mesh().export(off_path)
    print('Finished: {}'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run voxalization to off')
    parser.add_argument('-res', default=64, type=int)
    args = parser.parse_args()

    # ROOT = 'shapenet/data'

    unpackbits = True
    res = args.res
    min = -0.5
    max = 0.5

    voxel_path = '/home/brl/dataDisk2/wrw/dataset/ShapeNetDataSet_subset/ShapeNetCore.v1_processed/03001627/train/1007e20d5e811b308351982a6e40cf41/voxelization_64.npy'
    voxel_dir = os.path.dirname(voxel_path)
    create_voxel_off(voxel_dir)

    # p = Pool(mp.cpu_count())
    # p.map(create_voxel_off, glob.glob(ROOT + '/*/*/'))
