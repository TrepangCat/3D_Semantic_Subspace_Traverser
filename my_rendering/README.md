# rendering
Quick trimesh headless rendering for meshes. Supports depth map, normal map, ray to mesh intersection point and triangle ids.

Usage:

````Python
import trimesh
from matplotlib import pyplot as plt
from render import render_mesh
from render_utils import normalize_mesh

mesh = trimesh.load("path-to-file.obj")
normalize_mesh(mesh)

images, camera_poses = render_mesh(mesh, resolution=128, if_correct_normals=False)
triangle_ids, rendered_images, normal_maps, depth_images, p_images = images

plt.imshow(rendered_images[20])
````
