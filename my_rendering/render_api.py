import glob
import os

import trimesh
from skimage import io

from my_rendering.render import render_mesh
from my_rendering.render_utils import normalize_mesh

color_list = [
    [100, 200, 130],  # 渲染后，比较好看的绿色。
    [255, 130, 0],  # 渲染后，比较好看的橙色。
    [255,165,0],  # 渲染后，比较好看的黄色。
]


# camera_order 建议检查不同视角后，选择一个来用。
def render_trimesh_v1(mesh: trimesh.Trimesh, color=[100, 200, 130], image_resolution=512, camera_order=59):
    v, f = mesh.vertices, mesh.faces
    color = color
    mesh = trimesh.Trimesh(vertices=v, vertex_colors=color, faces=f)

    normalize_mesh(mesh)

    images, camera_poses = render_mesh(mesh, resolution=image_resolution, if_correct_normals=False,
                                       camera_order=camera_order)
    triangle_ids, rendered_images, normal_maps, depth_images, p_images = images

    return rendered_images[0]


if __name__ == '__main__':
    file_path = '/home/brl/data_disk_4t/code/code_test/edit3d-main/output/5ef54fa5d00dbdf8c8687ff9b0b4e4ac/1/init'
    obj_paths = glob.glob(os.path.join(file_path, '**/*.obj'), recursive=True)
    camera_order = 59

    for obj_path in obj_paths:
        mesh = trimesh.load(obj_path)
        v, f = mesh.vertices, mesh.faces
        # color = [3, 150, 255]  # 蓝色
        # color = [60,179,113]
        color = [100, 200, 130]  # 渲染后，比较好看的橙色。
        # color = [128, 128, 128]
        mesh = trimesh.Trimesh(vertices=v, vertex_colors=color, faces=f)
        normalize_mesh(mesh)

        images, camera_poses = render_mesh(mesh, resolution=512, if_correct_normals=False, camera_order=camera_order)
        triangle_ids, rendered_images, normal_maps, depth_images, p_images = images

        io.imsave(fname=obj_path[0:-4] + f'_cam{camera_order}.png', arr=rendered_images[0])
        # plt.imshow(rendered_images[20])
        # for i, image in enumerate(rendered_images):
        #     io.imsave(fname=obj_paths[0] + f'_render{i}.png', arr=image)

    print('finished')
