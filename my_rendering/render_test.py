import glob
import os

import trimesh
from matplotlib import pyplot as plt
from skimage import io

from render import render_mesh
from render_utils import normalize_mesh

if __name__ == '__main__':
    file_path = '/home/brl/data_disk_4t/code/stylegan_series/A_new_gan_project_july/editing_compare_with_DeepMetaHandle'
    obj_paths = glob.glob(os.path.join(file_path, '**/*.obj'), recursive=True)
    camera_order = 59
    camera_order = 64

    for obj_path in obj_paths:
        mesh = trimesh.load(obj_path)
        v, f = mesh.vertices, mesh.faces
        # color = [3, 150, 255]  # 蓝色
        # color = [60,179,113]
        color = [100,200,130]  # 渲染后，比较好看的绿色。
        mesh = trimesh.Trimesh(vertices=v, vertex_colors=color, faces=f)
        normalize_mesh(mesh)

        images, camera_poses = render_mesh(mesh, resolution=512, if_correct_normals=False, camera_order=camera_order)
        triangle_ids, rendered_images, normal_maps, depth_images, p_images = images

        io.imsave(fname=obj_path[0:-4] + f'_cam{camera_order}.png', arr=rendered_images[0])
        # plt.imshow(rendered_images[20])
        # for i, image in enumerate(rendered_images):
        #     io.imsave(fname=obj_paths[0] + f'_render{i}.png', arr=image)
