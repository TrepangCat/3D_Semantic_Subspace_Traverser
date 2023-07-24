import numpy as np
import trimesh
from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    dirs = [
        '/home/brl/dataDisk2/wrw/dataset/ShapeNetDataSet_subset/ShapeNetCore.v1_processed/03001627/train/origin'
    ]

    for dir in dirs:
        npz_list = glob(dir + '/**/boundary_SDF_samples200000.npz', recursive=True)

        for npz in tqdm(npz_list):
            samples = np.load(npz)

            pos = samples['pos']
            neg = samples['neg']

            neg_p = neg[:, 0:3]
            scale = abs(neg_p).max()

            pos[:, 0:3] = (pos[:, 0:3] / scale) * 0.999
            neg[:, 0:3] = (neg[:, 0:3] / scale) * 0.999  # 将neg的尺度，归一化到 [-0.999, 0.999]，而pos做同样的尺度变化，但不一定到[-0.999, 0.999]。

            np.savez(npz, pos=pos, neg=neg)


