# mostly done

import os
import numpy as np

import sys
sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
from py3d import *

project_dir = "/home/fregu856/exjobb/"
data_dir = project_dir + "data/kitti/object/training/"
img_dir = data_dir + "image_2/"
lidar_dir = data_dir + "velodyne/"

img_paths = []
label_paths = []
lidar_paths = []
img_names = os.listdir(img_dir)
for step, img_name in enumerate(img_names):
    img_id = img_name.split(".png")[0]

    img_path = img_dir + img_name
    img_paths.append(img_path)

    lidar_path = lidar_dir + img_id + ".bin"
    lidar_paths.append(lidar_path)

for lidar_path in lidar_paths:
    point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    print lidar_path
    print point_cloud.shape
    print point_cloud

    point_cloud_xyz = point_cloud[:, 0:3]

    pcd = PointCloud()
    pcd.points = Vector3dVector(point_cloud_xyz)
    draw_geometries([pcd])
