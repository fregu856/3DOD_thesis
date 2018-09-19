# camera-ready

import sys
sys.path.append("/root/3DOD_thesis/utils")
from kittiloader import LabelLoader2D3D, calibread, LabelLoader2D3D_sequence # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

import torch
import torch.utils.data

import os
import pickle
import numpy as np
import math
import cv2

def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def getBinNumber4(angle):
    # angle is assumed to be in [-pi, pi[

    if (angle >= -np.pi/4) and (angle < np.pi/4):
        bin_number = 0
    elif (angle >= np.pi/4) and (angle < 3*np.pi/4):
        bin_number = 1
    elif ((angle >= 3*np.pi/4) and (angle < np.pi)) or ((angle >= -np.pi) and (angle < -3*np.pi/4)):
        bin_number = 2
    elif (angle >= -3*np.pi/4) and (angle < -np.pi/4):
        bin_number = 3
    else:
        raise Exception("getBinNumber4: angle is not in [-pi, pi[")

    return bin_number

def getBinNumber(angle, NH):
    if NH == 4:
        return getBinNumber4(angle)
    else:
        raise Exception("getBinNumber: NH is not 4")

def getBinCenter(bin_number, NH):
    if NH == 4:
        bin_center = wrapToPi(bin_number*(np.pi/2))
    else:
        raise Exception("getBinCenter: NH is not 4")

    return bin_center

def getBinCenters(bin_numbers, NH):
    # bin_number has shape (m, n)

    if NH == 4:
        bin_centers = wrapToPi(bin_numbers*(np.pi/2))
    else:
        raise Exception("getBinCenters: NH is not 4")

    return bin_centers

class DatasetFrustumPointNetImgAugmentation(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, type, NH):
        self.img_dir = kitti_data_path + "/object/training/image_2/"
        self.label_dir = kitti_data_path + "/object/training/label_2/"
        self.calib_dir = kitti_data_path + "/object/training/calib/"
        self.lidar_dir = kitti_data_path + "/object/training/velodyne/"

        self.NH = NH

        with open(kitti_meta_path + "/%s_img_ids.pkl" % type, "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)

        with open(kitti_meta_path + "/kitti_centered_frustum_mean_xyz.pkl", "rb") as file: # (needed for python3)
            self.centered_frustum_mean_xyz = pickle.load(file)
            self.centered_frustum_mean_xyz = self.centered_frustum_mean_xyz.astype(np.float32)

        self.examples = []
        for img_id in img_ids:
            labels = LabelLoader2D3D(img_id, self.label_dir, ".txt", self.calib_dir, ".txt")
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["truncated"] < 0.5 and label_2d["class"] == "Car":
                    label["img_id"] = img_id
                    self.examples.append(label)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        lidar_path = self.lidar_dir + img_id + ".bin"
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        orig_point_cloud = point_cloud

        # remove points that are located behind the camera:
        point_cloud = point_cloud[point_cloud[:, 0] > 0, :]
        # remove points that are located too far away from the camera:
        point_cloud = point_cloud[point_cloud[:, 0] < 80, :]

        calib = calibread(self.calib_dir + img_id + ".txt")
        P2 = calib["P2"]
        Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
        R0_rect_orig = calib["R0_rect"]
        #
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        #
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

        # project the points onto the image plane (homogeneous coords):
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]

        # transform the points into (rectified) camera coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera

        label_2D = example["label_2D"]
        label_3D = example["label_3D"]

        bbox = label_2D["poly"]

        # img = cv2.imread(self.img_dir + img_id + ".png", -1)
        # img_with_bboxes = draw_2d_polys(img, [label_2D])
        # cv2.imwrite("test.png", img_with_bboxes)

        ########################################################################
        # frustum:
        ########################################################################
        u_min = bbox[0, 0] # (left)
        u_max = bbox[1, 0] # (rigth)
        v_min = bbox[0, 1] # (top)
        v_max = bbox[2, 1] # (bottom)

        ########################################################################
        # # # # augment the 2Dbbox:
        ########################################################################
        w = u_max - u_min
        h = v_max - v_min
        u_center = u_min + w/2.0
        v_center = v_min + h/2.0

        # translate the center by random distances sampled from
        # uniform[-0.1w, 0.1w] and uniform[-0.1h, 0.1h] in u,v directions:
        u_center = u_center + np.random.uniform(low=-0.1*w, high=0.1*w)
        v_center = v_center + np.random.uniform(low=-0.1*h, high=0.1*h)

        # randomly scale w and h by factor sampled from uniform[0.9, 1.1]:
        w = w*np.random.uniform(low=0.9, high=1.1)
        h = h*np.random.uniform(low=0.9, high=1.1)

        u_min = u_center - w/2.0
        u_max = u_center + w/2.0
        v_min = v_center - h/2.0
        v_max = v_center + h/2.0

        row_mask = np.logical_and(
                    np.logical_and(img_points[:, 0] >= u_min,
                                   img_points[:, 0] <= u_max),
                    np.logical_and(img_points[:, 1] >= v_min,
                                   img_points[:, 1] <= v_max))

        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :] # (needed only for visualization)
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]

        if frustum_point_cloud.shape[0] == 0:
            print (img_id)
            print (frustum_point_cloud.shape)
            return self.__getitem__(0)

        # randomly sample 1024 points in the frustum point cloud:
        if frustum_point_cloud.shape[0] < 1024:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=True)
        else:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=False)
        frustum_point_cloud_xyz = frustum_point_cloud_xyz[row_idx, :]
        frustum_point_cloud = frustum_point_cloud[row_idx, :]
        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        frustum_point_cloud_camera = frustum_point_cloud_camera[row_idx, :]
        # (the frustum point cloud now has exactly 1024 points)

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        # cv2.imshow("test", bbox_2d_img)
        # cv2.waitKey(0)

        ########################################################################
        # InstanceSeg ground truth:
        ########################################################################
        points = label_3D["points"]

        y_max = points[0, 1]
        y_min = points[4, 1]

        # D, A, B are consecutive corners of the rectangle (projection of the 3D bbox) in the x,z plane
        # the vectors AB and AD are orthogonal
        # a point P = (x, z) lies within the rectangle in the x,z plane iff:
        # (0 < AP dot AB < AB dot AB) && (0 < AP dot AD < AD dot AD)
        # (https://math.stackexchange.com/a/190373)

        A = np.array([points[0, 0], points[0, 2]])
        B = np.array([points[1, 0], points[1, 2]])
        D = np.array([points[3, 0], points[3, 2]])

        AB = B - A
        AD = D - A
        AB_dot_AB = np.dot(AB, AB)
        AD_dot_AD = np.dot(AD, AD)

        P = np.zeros((frustum_point_cloud_xyz_camera.shape[0], 2))
        P[:, 0] = frustum_point_cloud_xyz_camera[:, 0]
        P[:, 1] = frustum_point_cloud_xyz_camera[:, 2]

        AP = P - A
        AP_dot_AB = np.dot(AP, AB)
        AP_dot_AD = np.dot(AP, AD)

        row_mask = np.logical_and(
                    np.logical_and(frustum_point_cloud_xyz_camera[:, 1] >= y_min, frustum_point_cloud_xyz_camera[:, 1] <= y_max),
                    np.logical_and(np.logical_and(AP_dot_AB >= 0, AP_dot_AB <= AB_dot_AB),
                                   np.logical_and(AP_dot_AD >= 0, AP_dot_AD <= AD_dot_AD)))

        row_mask_gt = row_mask

        gt_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_mask, :] # (needed only for visualization)

        label_InstanceSeg = np.zeros((frustum_point_cloud.shape[0],), dtype=np.int64)
        label_InstanceSeg[row_mask] = 1 # (0: point is NOT part of the objet, 1: point is part of the object)

        ########################################################################
        # visualization of frustum and InstanceSeg ground truth:
        ########################################################################
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        #
        # # visualize the frustum points with ground truth as colored points:
        # frustum_pcd_camera = PointCloud()
        # frustum_pcd_camera.points = Vector3dVector(frustum_point_cloud_xyz_camera)
        # frustum_pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        # gt_pcd_camera = PointCloud()
        # gt_pcd_camera.points = Vector3dVector(gt_point_cloud_xyz_camera)
        # gt_pcd_camera.paint_uniform_color([0, 0, 1])
        # draw_geometries([gt_pcd_camera, frustum_pcd_camera])
        #
        # # visualize the frustum points and gt points as differently colored points in the full point cloud:
        # frustum_pcd_camera.paint_uniform_color([1, 0, 0])
        # pcd_camera = PointCloud()
        # pcd_camera.points = Vector3dVector(point_cloud_camera[:, 0:3])
        # pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        # draw_geometries([gt_pcd_camera, frustum_pcd_camera, pcd_camera])

        ########################################################################
        # normalize frustum point cloud:
        ########################################################################
        # get the 2dbbox center img point in hom. coords:
        u_center = u_min + (u_max - u_min)/2.0
        v_center = v_min + (v_max - v_min)/2.0
        center_img_point_hom = np.array([u_center, v_center, 1])

        # (more than one 3D point is projected onto the center image point, i.e,
        # the linear system of equations is under-determined and has inf number
        # of solutions. By using the pseudo-inverse, we obtain the least-norm sol)

        # get a point (the least-norm sol.) that projects onto the center image point, in hom. coords:
        P2_pseudo_inverse = np.linalg.pinv(P2) # (shape: (4, 3)) (P2 has shape (3, 4))
        point_hom = np.dot(P2_pseudo_inverse, center_img_point_hom)

        # hom --> normal coords:
        point = np.array(([point_hom[0]/point_hom[3], point_hom[1]/point_hom[3], point_hom[2]/point_hom[3]]))

        # if the point is behind the camera, switch to the mirror point in front of the camera:
        if point[2] < 0:
            point[0] = -point[0]
            point[2] = -point[2]

        # compute the angle of the point in the x-z plane: ((rectified) camera coords)
        frustum_angle = np.arctan2(point[0], point[2]) # (np.arctan2(x, z)) # (frustum_angle = 0: frustum is centered)

        # rotation_matrix to rotate points frustum_angle around the y axis (counter-clockwise):
        frustum_R = np.asarray([[np.cos(frustum_angle), 0, -np.sin(frustum_angle)],
                           [0, 1, 0],
                           [np.sin(frustum_angle), 0, np.cos(frustum_angle)]],
                           dtype='float32')

        # rotate the frustum point cloud to center it:
        centered_frustum_point_cloud_xyz_camera = np.dot(frustum_R, frustum_point_cloud_xyz_camera.T).T

        # subtract the centered frustum train xyz mean:
        centered_frustum_point_cloud_xyz_camera -= self.centered_frustum_mean_xyz

        centered_frustum_point_cloud_camera = frustum_point_cloud_camera
        centered_frustum_point_cloud_camera[:, 0:3] = centered_frustum_point_cloud_xyz_camera

        # # # # # # # # # # debug visualizations START:
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        #
        # frustum_pcd_camera = PointCloud()
        # frustum_pcd_camera.points = Vector3dVector(frustum_point_cloud_xyz_camera)
        # frustum_pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        #
        # centered_frustum_pcd_camera = PointCloud()
        # centered_frustum_pcd_camera.points = Vector3dVector(centered_frustum_point_cloud_xyz_camera)
        # centered_frustum_pcd_camera.paint_uniform_color([1, 0, 0])
        #
        # uncentered_frustum_point_cloud_xyz_camera = np.dot(np.linalg.inv(frustum_R), centered_frustum_point_cloud_xyz_camera.T).T
        # uncentered_frustum_pcd_camera = PointCloud()
        # uncentered_frustum_pcd_camera.points = Vector3dVector(uncentered_frustum_point_cloud_xyz_camera)
        # uncentered_frustum_pcd_camera.paint_uniform_color([0, 1, 0])
        # draw_geometries([centered_frustum_pcd_camera, uncentered_frustum_pcd_camera, frustum_pcd_camera])
        # # # # # # # # # # debug visualizations END:

        ########################################################################
        # randomly shift the frustum point cloud in the z direction:
        ########################################################################
        z_shift = np.random.uniform(low=-20, high=20)
        centered_frustum_point_cloud_camera[:, 2] -= z_shift

        ########################################################################
        # flip the frustum point cloud in the x-z plane with 0.5 prob:
        ########################################################################
        # # # # # # # # # # debug visualizations START:
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        # pcd = PointCloud()
        # pcd.points = Vector3dVector(centered_frustum_point_cloud_camera[:, 0:3])
        # pcd.paint_uniform_color([0.25, 0.25, 0.25])
        # draw_geometries([pcd])
        # # # # # # # # # # debug visualizations END:

        # get 0 or 1 with equal probability (indicating if the frustum should be flipped or not):
        flip = np.random.randint(low=0, high=2)

        # flip the frustum point cloud if flip == 1 (set all x values to -values):
        centered_frustum_point_cloud_camera[:, 0] = flip*(-centered_frustum_point_cloud_camera[:, 0]) + (1-flip)*centered_frustum_point_cloud_camera[:, 0]

        # # # # # # # # # # debug visualizations START:
        # pcd.points = Vector3dVector(centered_frustum_point_cloud_camera[:, 0:3])
        # pcd.paint_uniform_color([0.25, 0.25, 0.25])
        # draw_geometries([pcd])
        # # # # # # # # # # debug visualizations END:

        ########################################################################
        # flip the 2dbbox img crop if flip == 1:
        ########################################################################
        if flip == 1:
            bbox_2d_img = cv2.flip(bbox_2d_img, 1)
            # cv2.imshow("test", bbox_2d_img)
            # cv2.waitKey(0)

        ########################################################################
        # TNet ground truth:
        ########################################################################
        label_TNet = np.dot(frustum_R, label_3D["center"]) - self.centered_frustum_mean_xyz

        # flip the label if flip == 1 (set the x value to -value):
        label_TNet[0] = flip*(-label_TNet[0]) + (1-flip)*label_TNet[0]

        # adjust for the random shift:
        label_TNet[2] -= z_shift

        ########################################################################
        # BboxNet ground truth:
        ########################################################################
        centered_r_y = wrapToPi(label_3D['r_y'] - frustum_angle)
        # flip the angle if flip == 1:
        if flip == 1:
            centered_r_y = wrapToPi(np.pi - centered_r_y)

        bin_number = getBinNumber(centered_r_y, NH=self.NH)
        bin_center = getBinCenter(bin_number, NH=self.NH)
        residual = wrapToPi(centered_r_y - bin_center)

        label_BboxNet = np.zeros((11, ), dtype=np.float32) # ([x, y, z, h, w, l, r_y_bin_number, r_y_residual, r_y_bin_number_neighbor,r_y_residual_neighbor, h_mean, w_mean, l_mean])

        label_BboxNet[0:3] = np.dot(frustum_R, label_3D["center"]) - self.centered_frustum_mean_xyz
        # flip the label if flip == 1 (set the x value to -value):
        label_BboxNet[0] = flip*(-label_BboxNet[0]) + (1-flip)*label_BboxNet[0]
        # adjust for the random shift:
        label_BboxNet[2] -= z_shift

        label_BboxNet[3] = label_3D['h']
        label_BboxNet[4] = label_3D['w']
        label_BboxNet[5] = label_3D['l']
        label_BboxNet[6] = bin_number
        label_BboxNet[7] = residual
        label_BboxNet[8:] = self.mean_car_size

        ########################################################################
        # corner loss ground truth:
        ########################################################################
        Rmat = np.asarray([[math.cos(residual), 0, math.sin(residual)],
                           [0, 1, 0],
                           [-math.sin(residual), 0, math.cos(residual)]],
                           dtype='float32')

        center = label_BboxNet[0:3]
        l = label_3D['l']
        w = label_3D['w']
        h = label_3D['h']
        p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
        p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
        p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
        p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
        p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
        p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
        p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
        p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())
        label_corner = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
        label_corner_flipped = np.array([p2, p3, p0, p1, p6, p7, p4, p5])

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)
        ########################################################################

        centered_frustum_point_cloud_camera = torch.from_numpy(centered_frustum_point_cloud_camera) # (shape: (1024, 4))
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))
        label_InstanceSeg = torch.from_numpy(label_InstanceSeg) # (shape: (1024, ))
        label_TNet = torch.from_numpy(label_TNet) # (shape: (3, ))
        label_BboxNet = torch.from_numpy(label_BboxNet) # (shape: (11, ))
        label_corner = torch.from_numpy(label_corner) # (shape: (8, 3))
        label_corner_flipped = torch.from_numpy(label_corner_flipped) # (shape: (8, 3))

        return (centered_frustum_point_cloud_camera, bbox_2d_img, label_InstanceSeg, label_TNet, label_BboxNet, label_corner, label_corner_flipped)

    def __len__(self):
        return self.num_examples

# test = DatasetFrustumPointNetImgAugmentation("/home/fregu856/exjobb/data/kitti", "/home/fregu856/exjobb/data/kitti/meta", type="train", NH=4)
# for i in range(10):
#     _ = test.__getitem__(i)

class EvalDatasetFrustumPointNetImg(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, type, NH):
        self.img_dir = kitti_data_path + "/object/training/image_2/"
        self.label_dir = kitti_data_path + "/object/training/label_2/"
        self.calib_dir = kitti_data_path + "/object/training/calib/"
        self.lidar_dir = kitti_data_path + "/object/training/velodyne/"

        self.NH = NH

        with open(kitti_meta_path + "/%s_img_ids.pkl" % type, "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)

        with open(kitti_meta_path + "/kitti_centered_frustum_mean_xyz.pkl", "rb") as file: # (needed for python3)
            self.centered_frustum_mean_xyz = pickle.load(file)
            self.centered_frustum_mean_xyz = self.centered_frustum_mean_xyz.astype(np.float32)

        self.examples = []
        for img_id in img_ids:
            labels = LabelLoader2D3D(img_id, self.label_dir, ".txt", self.calib_dir, ".txt")
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["truncated"] < 0.5 and label_2d["class"] == "Car":
                    label["img_id"] = img_id
                    self.examples.append(label)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        lidar_path = self.lidar_dir + img_id + ".bin"
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        orig_point_cloud = point_cloud

        # remove points that are located behind the camera:
        point_cloud = point_cloud[point_cloud[:, 0] > 0, :]
        # remove points that are located too far away from the camera:
        point_cloud = point_cloud[point_cloud[:, 0] < 80, :]

        calib = calibread(self.calib_dir + img_id + ".txt")
        P2 = calib["P2"]
        Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
        R0_rect_orig = calib["R0_rect"]
        #
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        #
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

        # project the points onto the image plane (homogeneous coords):
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]

        # transform the points into (rectified) camera coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera

        label_2D = example["label_2D"]
        label_3D = example["label_3D"]

        bbox = label_2D["poly"]

        # img = cv2.imread(self.img_dir + img_id + ".png", -1)
        # img_with_bboxes = draw_2d_polys(img, [label_2D])
        # cv2.imwrite("test.png", img_with_bboxes)

        ########################################################################
        # frustum:
        ########################################################################
        u_min = bbox[0, 0] # (left)
        u_max = bbox[1, 0] # (rigth)
        v_min = bbox[0, 1] # (top)
        v_max = bbox[2, 1] # (bottom)

        u_min_expanded = u_min #- (u_max-u_min)*0.05
        u_max_expanded = u_max #+ (u_max-u_min)*0.05
        v_min_expanded = v_min #- (v_max-v_min)*0.05
        v_max_expanded = v_max #+ (v_max-v_min)*0.05
        input_2Dbbox = np.array([u_min_expanded, u_max_expanded, v_min_expanded, v_max_expanded])

        row_mask = np.logical_and(
                    np.logical_and(img_points[:, 0] >= u_min_expanded,
                                   img_points[:, 0] <= u_max_expanded),
                    np.logical_and(img_points[:, 1] >= v_min_expanded,
                                   img_points[:, 1] <= v_max_expanded))

        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :] # (needed only for visualization)
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]

        if frustum_point_cloud.shape[0] == 0:
            print (img_id)
            print (frustum_point_cloud.shape)
            return self.__getitem__(0)

        # randomly sample 1024 points in the frustum point cloud:
        if frustum_point_cloud.shape[0] < 1024:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=True)
        else:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=False)
        frustum_point_cloud_xyz = frustum_point_cloud_xyz[row_idx, :]
        frustum_point_cloud = frustum_point_cloud[row_idx, :]
        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        frustum_point_cloud_camera = frustum_point_cloud_camera[row_idx, :]
        # (the frustum point cloud now has exactly 1024 points)

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        # cv2.imshow("test", bbox_2d_img)
        # cv2.waitKey(0)

        ########################################################################
        # InstanceSeg ground truth:
        ########################################################################
        points = label_3D["points"]

        y_max = points[0, 1]
        y_min = points[4, 1]

        # D, A, B are consecutive corners of the rectangle (projection of the 3D bbox) in the x,z plane
        # the vectors AB and AD are orthogonal
        # a point P = (x, z) lies within the rectangle in the x,z plane iff:
        # (0 < AP dot AB < AB dot AB) && (0 < AP dot AD < AD dot AD)
        # (https://math.stackexchange.com/a/190373)

        A = np.array([points[0, 0], points[0, 2]])
        B = np.array([points[1, 0], points[1, 2]])
        D = np.array([points[3, 0], points[3, 2]])

        AB = B - A
        AD = D - A
        AB_dot_AB = np.dot(AB, AB)
        AD_dot_AD = np.dot(AD, AD)

        P = np.zeros((frustum_point_cloud_xyz_camera.shape[0], 2))
        P[:, 0] = frustum_point_cloud_xyz_camera[:, 0]
        P[:, 1] = frustum_point_cloud_xyz_camera[:, 2]

        AP = P - A
        AP_dot_AB = np.dot(AP, AB)
        AP_dot_AD = np.dot(AP, AD)

        row_mask = np.logical_and(
                    np.logical_and(frustum_point_cloud_xyz_camera[:, 1] >= y_min, frustum_point_cloud_xyz_camera[:, 1] <= y_max),
                    np.logical_and(np.logical_and(AP_dot_AB >= 0, AP_dot_AB <= AB_dot_AB),
                                   np.logical_and(AP_dot_AD >= 0, AP_dot_AD <= AD_dot_AD)))

        gt_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_mask, :] # (needed only for visualization)

        label_InstanceSeg = np.zeros((frustum_point_cloud.shape[0],), dtype=np.int64)
        label_InstanceSeg[row_mask] = 1 # (0: point is NOT part of the objet, 1: point is part of the object)

        ########################################################################
        # visualization of frustum and InstanceSeg ground truth:
        ########################################################################
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        #
        # # visualize the frustum points with ground truth as colored points:
        # frustum_pcd_camera = PointCloud()
        # frustum_pcd_camera.points = Vector3dVector(frustum_point_cloud_xyz_camera)
        # frustum_pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        # gt_pcd_camera = PointCloud()
        # gt_pcd_camera.points = Vector3dVector(gt_point_cloud_xyz_camera)
        # gt_pcd_camera.paint_uniform_color([0, 0, 1])
        # draw_geometries([gt_pcd_camera, frustum_pcd_camera])

        # # visualize the frustum points and gt points as differently colored points in the full point cloud:
        # frustum_pcd_camera.paint_uniform_color([1, 0, 0])
        # pcd_camera = PointCloud()
        # pcd_camera.points = Vector3dVector(orig_point_cloud_camera[:, 0:3])
        # pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        # draw_geometries([gt_pcd_camera, frustum_pcd_camera, pcd_camera])

        ########################################################################
        # normalize frustum point cloud:
        ########################################################################
        # get the 2dbbox center img point in hom. coords:
        u_center = u_min_expanded + (u_max_expanded - u_min_expanded)/2.0
        v_center = v_min_expanded + (v_max_expanded - v_min_expanded)/2.0
        center_img_point_hom = np.array([u_center, v_center, 1])

        # (more than one 3D point is projected onto the center image point, i.e,
        # the linear system of equations is under-determined and has inf number
        # of solutions. By using the pseudo-inverse, we obtain the least-norm sol)

        # get a point (the least-norm sol.) that projects onto the center image point, in hom. coords:
        P2_pseudo_inverse = np.linalg.pinv(P2) # (shape: (4, 3)) (P2 has shape (3, 4))
        point_hom = np.dot(P2_pseudo_inverse, center_img_point_hom)

        # hom --> normal coords:
        point = np.array(([point_hom[0]/point_hom[3], point_hom[1]/point_hom[3], point_hom[2]/point_hom[3]]))

        # if the point is behind the camera, switch to the mirror point in front of the camera:
        if point[2] < 0:
            point[0] = -point[0]
            point[2] = -point[2]

        # compute the angle of the point in the x-z plane: ((rectified) camera coords)
        frustum_angle = np.arctan2(point[0], point[2]) # (np.arctan2(x, z)) # (frustum_angle = 0: frustum is centered)

        # rotation_matrix to rotate points frustum_angle around the y axis (counter-clockwise):
        frustum_R = np.asarray([[np.cos(frustum_angle), 0, -np.sin(frustum_angle)],
                           [0, 1, 0],
                           [np.sin(frustum_angle), 0, np.cos(frustum_angle)]],
                           dtype='float32')

        # rotate the frustum point cloud to center it:
        centered_frustum_point_cloud_xyz_camera = np.dot(frustum_R, frustum_point_cloud_xyz_camera.T).T

        # subtract the centered frustum train xyz mean:
        centered_frustum_point_cloud_xyz_camera -= self.centered_frustum_mean_xyz

        centered_frustum_point_cloud_camera = frustum_point_cloud_camera
        centered_frustum_point_cloud_camera[:, 0:3] = centered_frustum_point_cloud_xyz_camera

        # # # # # # # # # # debug visualizations START:
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        #
        # frustum_pcd_camera = PointCloud()
        # frustum_pcd_camera.points = Vector3dVector(frustum_point_cloud_xyz_camera)
        # frustum_pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        #
        # centered_frustum_pcd_camera = PointCloud()
        # centered_frustum_pcd_camera.points = Vector3dVector(centered_frustum_point_cloud_xyz_camera)
        # centered_frustum_pcd_camera.paint_uniform_color([1, 0, 0])
        # draw_geometries([centered_frustum_pcd_camera, frustum_pcd_camera])
        # # # # # # # # # # debug visualizations END:

        ########################################################################
        # TNet ground truth:
        ########################################################################
        label_TNet = np.dot(frustum_R, label_3D["center"]) - self.centered_frustum_mean_xyz

        # # # # # # # # # # debug visualizations START:
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        #
        # # visualize InstanceSeg GT and 3dbbox center GT:
        # frustum_pcd_camera = PointCloud()
        # frustum_pcd_camera.points = Vector3dVector(frustum_point_cloud_xyz_camera)
        # frustum_pcd_camera.paint_uniform_color([0.25, 0.25, 0.25cd
        # gt_pcd_camera = PointCloud()
        # gt_pcd_camera.points = Vector3dVector(gt_point_cloud_xyz_camera)
        # gt_pcd_camera.paint_uniform_color([0, 0, 1])
        # center_gt_pcd_camera = PointCloud()
        # center_gt_pcd_camera.points = Vector3dVector(np.array([label_TNet]))
        # center_gt_pcd_camera.paint_uniform_color([0, 1, 0])
        # draw_geometries([center_gt_pcd_camera, gt_pcd_camera, frustum_pcd_camera])
        # # # # # # # # # # debug visualizations END:

        ########################################################################
        # BboxNet ground truth:
        ########################################################################
        centered_r_y = wrapToPi(label_3D['r_y'] - frustum_angle)
        bin_number = getBinNumber(centered_r_y, NH=self.NH)
        bin_center = getBinCenter(bin_number, NH=self.NH)
        residual = wrapToPi(centered_r_y - bin_center)

        label_BboxNet = np.zeros((11, ), dtype=np.float32) # ([x, y, z, h, w, l, r_y_bin_number, r_y_residual, r_y_bin_number_neighbor,r_y_residual_neighbor, h_mean, w_mean, l_mean])

        label_BboxNet[0:3] = np.dot(frustum_R, label_3D["center"]) - self.centered_frustum_mean_xyz
        label_BboxNet[3] = label_3D['h']
        label_BboxNet[4] = label_3D['w']
        label_BboxNet[5] = label_3D['l']
        label_BboxNet[6] = bin_number
        label_BboxNet[7] = residual
        label_BboxNet[8:] = self.mean_car_size

        ########################################################################
        # corner loss ground truth:
        ########################################################################
        Rmat = np.asarray([[math.cos(residual), 0, math.sin(residual)],
                           [0, 1, 0],
                           [-math.sin(residual), 0, math.cos(residual)]],
                           dtype='float32')

        center = label_BboxNet[0:3]
        l = label_3D['l']
        w = label_3D['w']
        h = label_3D['h']
        p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
        p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
        p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
        p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
        p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
        p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
        p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
        p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())
        label_corner = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
        label_corner_flipped = np.array([p2, p3, p0, p1, p6, p7, p4, p5])

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)
        ########################################################################

        centered_frustum_point_cloud_camera = torch.from_numpy(centered_frustum_point_cloud_camera) # (shape: (1024, 4))
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))
        label_InstanceSeg = torch.from_numpy(label_InstanceSeg) # (shape: (1024, ))
        label_TNet = torch.from_numpy(label_TNet) # (shape: (3, ))
        label_BboxNet = torch.from_numpy(label_BboxNet) # (shape: (11, ))
        label_corner = torch.from_numpy(label_corner) # (shape: (8, 3))
        label_corner_flipped = torch.from_numpy(label_corner_flipped) # (shape: (8, 3))

        return (centered_frustum_point_cloud_camera, bbox_2d_img, label_InstanceSeg, label_TNet, label_BboxNet, label_corner, label_corner_flipped, img_id, input_2Dbbox, frustum_R, frustum_angle, self.centered_frustum_mean_xyz)

    def __len__(self):
        return self.num_examples

class EvalSequenceDatasetFrustumPointNetImg(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, NH, sequence):
        self.img_dir = kitti_data_path + "/tracking/training/image_02/" + sequence + "/"
        self.lidar_dir = kitti_data_path + "/tracking/training/velodyne/" + sequence + "/"
        self.label_path = kitti_data_path + "/tracking/training/label_02/" + sequence + ".txt"
        self.calib_path = kitti_meta_path + "/tracking/training/calib/" + sequence + ".txt" # NOTE! NOTE! the data format for the calib files was sliightly different for tracking, so I manually modifed the 20 files and saved them in the kitti_meta folder

        self.NH = NH

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)

        with open(kitti_meta_path + "/kitti_centered_frustum_mean_xyz.pkl", "rb") as file: # (needed for python3)
            self.centered_frustum_mean_xyz = pickle.load(file)
            self.centered_frustum_mean_xyz = self.centered_frustum_mean_xyz.astype(np.float32)

        img_ids = []
        img_names = os.listdir(self.img_dir)
        for img_name in img_names:
            img_id = img_name.split(".png")[0]
            img_ids.append(img_id)

        self.examples = []
        for img_id in img_ids:
            if img_id.lstrip('0') == '':
                img_id_float = 0.0
            else:
                img_id_float = float(img_id.lstrip('0'))

            labels = LabelLoader2D3D_sequence(img_id, img_id_float, self.label_path, self.calib_path)

            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["truncated"] < 0.5 and label_2d["class"] == "Car":
                    label["img_id"] = img_id
                    self.examples.append(label)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        lidar_path = self.lidar_dir + img_id + ".bin"
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        orig_point_cloud = point_cloud

        # remove points that are located behind the camera: # TODO! should we allow some points behind the camera when we actually have amodal 2D bboxes??
        point_cloud = point_cloud[point_cloud[:, 0] > 0, :]
        # remove points that are located too far away from the camera:
        point_cloud = point_cloud[point_cloud[:, 0] < 80, :]

        calib = calibread(self.calib_path)
        P2 = calib["P2"]
        Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
        R0_rect_orig = calib["R0_rect"]
        #
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        #
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

        # project the points onto the image plane (homogeneous coords):
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]

        # transform the points into (rectified) camera coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera

        label_2D = example["label_2D"]
        label_3D = example["label_3D"]

        bbox = label_2D["poly"]

        # img = cv2.imread(self.img_dir + img_id + ".png", -1)
        # img_with_bboxes = draw_2d_polys(img, [label_2D])
        # cv2.imwrite("test.png", img_with_bboxes)

        ########################################################################
        # frustum:
        ########################################################################
        u_min = bbox[0, 0] # (left)
        u_max = bbox[1, 0] # (rigth)
        v_min = bbox[0, 1] # (top)
        v_max = bbox[2, 1] # (bottom)

        u_min_expanded = u_min #- (u_max-u_min)*0.05
        u_max_expanded = u_max #+ (u_max-u_min)*0.05
        v_min_expanded = v_min #- (v_max-v_min)*0.05
        v_max_expanded = v_max #+ (v_max-v_min)*0.05
        input_2Dbbox = np.array([u_min_expanded, u_max_expanded, v_min_expanded, v_max_expanded])

        row_mask = np.logical_and(
                    np.logical_and(img_points[:, 0] >= u_min_expanded,
                                   img_points[:, 0] <= u_max_expanded),
                    np.logical_and(img_points[:, 1] >= v_min_expanded,
                                   img_points[:, 1] <= v_max_expanded))

        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :] # (needed only for visualization)
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]

        if frustum_point_cloud.shape[0] == 0:
            print (img_id)
            print (frustum_point_cloud.shape)
            return self.__getitem__(0)

        # randomly sample 1024 points in the frustum point cloud:
        if frustum_point_cloud.shape[0] < 1024:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=True)
        else:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=False)
        frustum_point_cloud_xyz = frustum_point_cloud_xyz[row_idx, :]
        frustum_point_cloud = frustum_point_cloud[row_idx, :]
        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        frustum_point_cloud_camera = frustum_point_cloud_camera[row_idx, :]
        # (the frustum point cloud now has exactly 1024 points)

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        # cv2.imshow("test", bbox_2d_img)
        # cv2.waitKey(0)

        ########################################################################
        # InstanceSeg ground truth:
        ########################################################################
        points = label_3D["points"]

        y_max = points[0, 1]
        y_min = points[4, 1]

        # D, A, B are consecutive corners of the rectangle (projection of the 3D bbox) in the x,z plane
        # the vectors AB and AD are orthogonal
        # a point P = (x, z) lies within the rectangle in the x,z plane iff:
        # (0 < AP dot AB < AB dot AB) && (0 < AP dot AD < AD dot AD)
        # (https://math.stackexchange.com/a/190373)

        A = np.array([points[0, 0], points[0, 2]])
        B = np.array([points[1, 0], points[1, 2]])
        D = np.array([points[3, 0], points[3, 2]])

        AB = B - A
        AD = D - A
        AB_dot_AB = np.dot(AB, AB)
        AD_dot_AD = np.dot(AD, AD)

        P = np.zeros((frustum_point_cloud_xyz_camera.shape[0], 2))
        P[:, 0] = frustum_point_cloud_xyz_camera[:, 0]
        P[:, 1] = frustum_point_cloud_xyz_camera[:, 2]

        AP = P - A
        AP_dot_AB = np.dot(AP, AB)
        AP_dot_AD = np.dot(AP, AD)

        row_mask = np.logical_and(
                    np.logical_and(frustum_point_cloud_xyz_camera[:, 1] >= y_min, frustum_point_cloud_xyz_camera[:, 1] <= y_max),
                    np.logical_and(np.logical_and(AP_dot_AB >= 0, AP_dot_AB <= AB_dot_AB),
                                   np.logical_and(AP_dot_AD >= 0, AP_dot_AD <= AD_dot_AD)))

        gt_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_mask, :] # (needed only for visualization)

        label_InstanceSeg = np.zeros((frustum_point_cloud.shape[0],), dtype=np.int64)
        label_InstanceSeg[row_mask] = 1 # (0: point is NOT part of the objet, 1: point is part of the object)

        ########################################################################
        # visualization of frustum and InstanceSeg ground truth:
        ########################################################################
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        #
        # # visualize the frustum points with ground truth as colored points:
        # frustum_pcd_camera = PointCloud()
        # frustum_pcd_camera.points = Vector3dVector(frustum_point_cloud_xyz_camera)
        # frustum_pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        # gt_pcd_camera = PointCloud()
        # gt_pcd_camera.points = Vector3dVector(gt_point_cloud_xyz_camera)
        # gt_pcd_camera.paint_uniform_color([0, 0, 1])
        # draw_geometries([gt_pcd_camera, frustum_pcd_camera])

        # # visualize the frustum points and gt points as differently colored points in the full point cloud:
        # frustum_pcd_camera.paint_uniform_color([1, 0, 0])
        # pcd_camera = PointCloud()
        # pcd_camera.points = Vector3dVector(orig_point_cloud_camera[:, 0:3])
        # pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        # draw_geometries([gt_pcd_camera, frustum_pcd_camera, pcd_camera])

        ########################################################################
        # normalize frustum point cloud:
        ########################################################################
        # get the 2dbbox center img point in hom. coords:
        u_center = u_min_expanded + (u_max_expanded - u_min_expanded)/2.0
        v_center = v_min_expanded + (v_max_expanded - v_min_expanded)/2.0
        center_img_point_hom = np.array([u_center, v_center, 1])

        # (more than one 3D point is projected onto the center image point, i.e,
        # the linear system of equations is under-determined and has inf number
        # of solutions. By using the pseudo-inverse, we obtain the least-norm sol)

        # get a point (the least-norm sol.) that projects onto the center image point, in hom. coords:
        P2_pseudo_inverse = np.linalg.pinv(P2) # (shape: (4, 3)) (P2 has shape (3, 4))
        point_hom = np.dot(P2_pseudo_inverse, center_img_point_hom)

        # hom --> normal coords:
        point = np.array(([point_hom[0]/point_hom[3], point_hom[1]/point_hom[3], point_hom[2]/point_hom[3]]))

        # if the point is behind the camera, switch to the mirror point in front of the camera:
        if point[2] < 0:
            point[0] = -point[0]
            point[2] = -point[2]

        # compute the angle of the point in the x-z plane: ((rectified) camera coords)
        frustum_angle = np.arctan2(point[0], point[2]) # (np.arctan2(x, z)) # (frustum_angle = 0: frustum is centered)

        # rotation_matrix to rotate points frustum_angle around the y axis (counter-clockwise):
        frustum_R = np.asarray([[np.cos(frustum_angle), 0, -np.sin(frustum_angle)],
                           [0, 1, 0],
                           [np.sin(frustum_angle), 0, np.cos(frustum_angle)]],
                           dtype='float32')

        # rotate the frustum point cloud to center it:
        centered_frustum_point_cloud_xyz_camera = np.dot(frustum_R, frustum_point_cloud_xyz_camera.T).T

        # subtract the centered frustum train xyz mean:
        centered_frustum_point_cloud_xyz_camera -= self.centered_frustum_mean_xyz

        centered_frustum_point_cloud_camera = frustum_point_cloud_camera
        centered_frustum_point_cloud_camera[:, 0:3] = centered_frustum_point_cloud_xyz_camera

        # # # # # # # # # # debug visualizations START:
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        #
        # frustum_pcd_camera = PointCloud()
        # frustum_pcd_camera.points = Vector3dVector(frustum_point_cloud_xyz_camera)
        # frustum_pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        #
        # centered_frustum_pcd_camera = PointCloud()
        # centered_frustum_pcd_camera.points = Vector3dVector(centered_frustum_point_cloud_xyz_camera)
        # centered_frustum_pcd_camera.paint_uniform_color([1, 0, 0])
        # draw_geometries([centered_frustum_pcd_camera, frustum_pcd_camera])
        # # # # # # # # # # debug visualizations END:

        ########################################################################
        # TNet ground truth:
        ########################################################################
        label_TNet = np.dot(frustum_R, label_3D["center"]) - self.centered_frustum_mean_xyz

        # # # # # # # # # # debug visualizations START:
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        #
        # # visualize InstanceSeg GT and 3dbbox center GT:
        # frustum_pcd_camera = PointCloud()
        # frustum_pcd_camera.points = Vector3dVector(frustum_point_cloud_xyz_camera)
        # frustum_pcd_camera.paint_uniform_color([0.25, 0.25, 0.25cd
        # gt_pcd_camera = PointCloud()
        # gt_pcd_camera.points = Vector3dVector(gt_point_cloud_xyz_camera)
        # gt_pcd_camera.paint_uniform_color([0, 0, 1])
        # center_gt_pcd_camera = PointCloud()
        # center_gt_pcd_camera.points = Vector3dVector(np.array([label_TNet]))
        # center_gt_pcd_camera.paint_uniform_color([0, 1, 0])
        # draw_geometries([center_gt_pcd_camera, gt_pcd_camera, frustum_pcd_camera])
        # # # # # # # # # # debug visualizations END:

        ########################################################################
        # BboxNet ground truth:
        ########################################################################
        centered_r_y = wrapToPi(label_3D['r_y'] - frustum_angle)
        bin_number = getBinNumber(centered_r_y, NH=self.NH)
        bin_center = getBinCenter(bin_number, NH=self.NH)
        residual = wrapToPi(centered_r_y - bin_center)

        label_BboxNet = np.zeros((11, ), dtype=np.float32) # ([x, y, z, h, w, l, r_y_bin_number, r_y_residual, r_y_bin_number_neighbor,r_y_residual_neighbor, h_mean, w_mean, l_mean])

        label_BboxNet[0:3] = np.dot(frustum_R, label_3D["center"]) - self.centered_frustum_mean_xyz
        label_BboxNet[3] = label_3D['h']
        label_BboxNet[4] = label_3D['w']
        label_BboxNet[5] = label_3D['l']
        label_BboxNet[6] = bin_number
        label_BboxNet[7] = residual
        label_BboxNet[8:] = self.mean_car_size

        ########################################################################
        # corner loss ground truth:
        ########################################################################
        Rmat = np.asarray([[math.cos(residual), 0, math.sin(residual)],
                           [0, 1, 0],
                           [-math.sin(residual), 0, math.cos(residual)]],
                           dtype='float32')

        center = label_BboxNet[0:3]
        l = label_3D['l']
        w = label_3D['w']
        h = label_3D['h']
        p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
        p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
        p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
        p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
        p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
        p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
        p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
        p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())
        label_corner = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
        label_corner_flipped = np.array([p2, p3, p0, p1, p6, p7, p4, p5])

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)
        ########################################################################

        centered_frustum_point_cloud_camera = torch.from_numpy(centered_frustum_point_cloud_camera) # (shape: (1024, 4))
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))
        label_InstanceSeg = torch.from_numpy(label_InstanceSeg) # (shape: (1024, ))
        label_TNet = torch.from_numpy(label_TNet) # (shape: (3, ))
        label_BboxNet = torch.from_numpy(label_BboxNet) # (shape: (11, ))
        label_corner = torch.from_numpy(label_corner) # (shape: (8, 3))
        label_corner_flipped = torch.from_numpy(label_corner_flipped) # (shape: (8, 3))

        return (centered_frustum_point_cloud_camera, bbox_2d_img, label_InstanceSeg, label_TNet, label_BboxNet, label_corner, label_corner_flipped, img_id, input_2Dbbox, frustum_R, frustum_angle, self.centered_frustum_mean_xyz)

    def __len__(self):
        return self.num_examples

class DatasetKittiTestSequence(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, NH, sequence):
        self.img_dir = kitti_data_path + "/tracking/testing/image_02/" + sequence + "/"
        self.lidar_dir = kitti_data_path + "/tracking/testing/velodyne/" + sequence + "/"
        self.calib_path = kitti_meta_path + "/tracking/testing/calib/" + sequence + ".txt" # NOTE! NOTE! the data format for the calib files was sliightly different for tracking, so I manually modifed the 28 files and saved them in the kitti_meta folder
        self.detections_2d_path = kitti_meta_path + "/tracking/testing/2d_detections/" + sequence + "/inferResult_1.txt"

        self.NH = NH

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)

        with open(kitti_meta_path + "/kitti_centered_frustum_mean_xyz.pkl", "rb") as file: # (needed for python3)
            self.centered_frustum_mean_xyz = pickle.load(file)
            self.centered_frustum_mean_xyz = self.centered_frustum_mean_xyz.astype(np.float32)

        self.examples = []
        with open(self.detections_2d_path) as file:
            # line format: img_id, img_height, img_width, class, u_min, v_min, u_max, v_max, confidence score, distance estimate
            for line in file:
                values = line.split()
                object_class = float(values[3])
                if object_class == 1: # (1: Car)
                    img_id = values[0]
                    u_min = float(values[4])
                    v_min = float(values[5])
                    u_max = float(values[6])
                    v_max = float(values[7])
                    score_2d = float(values[8])

                    detection_2d = {}
                    detection_2d["u_min"] = u_min
                    detection_2d["v_min"] = v_min
                    detection_2d["u_max"] = u_max
                    detection_2d["v_max"] = v_max
                    detection_2d["score_2d"] = score_2d
                    detection_2d["img_id"] = img_id

                    self.examples.append(detection_2d)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        lidar_path = self.lidar_dir + img_id + ".bin"
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        orig_point_cloud = point_cloud

        # remove points that are located behind the camera:
        point_cloud = point_cloud[point_cloud[:, 0] > -5, :]
        # remove points that are located too far away from the camera:
        point_cloud = point_cloud[point_cloud[:, 0] < 80, :]

        calib = calibread(self.calib_path)
        P2 = calib["P2"]
        Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
        R0_rect_orig = calib["R0_rect"]
        #
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        #
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

        # project the points onto the image plane (homogeneous coords):
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]

        # transform the points into (rectified) camera coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera

        ########################################################################
        # frustum:
        ########################################################################
        u_min = example["u_min"] # (left)
        u_max = example["u_max"] # (rigth)
        v_min = example["v_min"] # (top)
        v_max = example["v_max"] # (bottom)

        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])

        row_mask = np.logical_and(
                    np.logical_and(img_points[:, 0] >= u_min,
                                   img_points[:, 0] <= u_max),
                    np.logical_and(img_points[:, 1] >= v_min,
                                   img_points[:, 1] <= v_max))

        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :] # (needed only for visualization)
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]

        empty_frustum_flag = 0
        if frustum_point_cloud.shape[0] == 0:
            empty_frustum_flag = 1
            frustum_point_cloud = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz = np.zeros((1024, 3), dtype=np.float32)
            frustum_point_cloud_camera = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz_camera = np.zeros((1024, 3), dtype=np.float32)

        # randomly sample 1024 points in the frustum point cloud:
        if frustum_point_cloud.shape[0] < 1024:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=True)
        else:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=False)
        frustum_point_cloud_xyz = frustum_point_cloud_xyz[row_idx, :]
        frustum_point_cloud = frustum_point_cloud[row_idx, :]
        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        frustum_point_cloud_camera = frustum_point_cloud_camera[row_idx, :]
        # (the frustum point cloud now has exactly 1024 points)

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        # cv2.imshow("test", bbox_2d_img)
        # cv2.waitKey(0)

        ########################################################################
        # normalize frustum point cloud:
        ########################################################################
        # get the 2dbbox center img point in hom. coords:
        u_center = u_min + (u_max - u_min)/2.0
        v_center = v_min + (v_max - v_min)/2.0
        center_img_point_hom = np.array([u_center, v_center, 1])

        # (more than one 3D point is projected onto the center image point, i.e,
        # the linear system of equations is under-determined and has inf number
        # of solutions. By using the pseudo-inverse, we obtain the least-norm sol)

        # get a point (the least-norm sol.) that projects onto the center image point, in hom. coords:
        P2_pseudo_inverse = np.linalg.pinv(P2) # (shape: (4, 3)) (P2 has shape (3, 4))
        point_hom = np.dot(P2_pseudo_inverse, center_img_point_hom)

        # hom --> normal coords:
        point = np.array(([point_hom[0]/point_hom[3], point_hom[1]/point_hom[3], point_hom[2]/point_hom[3]]))

        # if the point is behind the camera, switch to the mirror point in front of the camera:
        if point[2] < 0:
            point[0] = -point[0]
            point[2] = -point[2]

        # compute the angle of the point in the x-z plane: ((rectified) camera coords)
        frustum_angle = np.arctan2(point[0], point[2]) # (np.arctan2(x, z)) # (frustum_angle = 0: frustum is centered)

        # rotation_matrix to rotate points frustum_angle around the y axis (counter-clockwise):
        frustum_R = np.asarray([[np.cos(frustum_angle), 0, -np.sin(frustum_angle)],
                           [0, 1, 0],
                           [np.sin(frustum_angle), 0, np.cos(frustum_angle)]],
                           dtype='float32')

        # rotate the frustum point cloud to center it:
        centered_frustum_point_cloud_xyz_camera = np.dot(frustum_R, frustum_point_cloud_xyz_camera.T).T

        # subtract the centered frustum train xyz mean:
        centered_frustum_point_cloud_xyz_camera -= self.centered_frustum_mean_xyz

        centered_frustum_point_cloud_camera = frustum_point_cloud_camera
        centered_frustum_point_cloud_camera[:, 0:3] = centered_frustum_point_cloud_xyz_camera

        # # # # # # # # # # debug visualizations START:
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        # frustum_pcd_camera = PointCloud()
        # frustum_pcd_camera.points = Vector3dVector(frustum_point_cloud_xyz_camera)
        # frustum_pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        # centered_frustum_pcd_camera = PointCloud()
        # centered_frustum_pcd_camera.points = Vector3dVector(centered_frustum_point_cloud_xyz_camera)
        # centered_frustum_pcd_camera.paint_uniform_color([1, 0, 0])
        # draw_geometries([centered_frustum_pcd_camera, frustum_pcd_camera])
        # # # # # # # # # # debug visualizations END:

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)

        centered_frustum_point_cloud_camera = torch.from_numpy(centered_frustum_point_cloud_camera) # (shape: (1024, 4))
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))

        return (centered_frustum_point_cloud_camera, bbox_2d_img, img_id, input_2Dbbox, frustum_R, frustum_angle, empty_frustum_flag, self.centered_frustum_mean_xyz, self.mean_car_size)

    def __len__(self):
        return self.num_examples

class DatasetKittiTest(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, NH):
        self.img_dir = kitti_data_path + "/object/testing/image_2/"
        self.calib_dir = kitti_data_path + "/object/testing/calib/"
        self.lidar_dir = kitti_data_path + "/object/testing/velodyne/"
        self.detections_2d_dir = kitti_meta_path + "/object/testing/2d_detections/"

        self.NH = NH

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)

        with open(kitti_meta_path + "/kitti_centered_frustum_mean_xyz.pkl", "rb") as file: # (needed for python3)
            self.centered_frustum_mean_xyz = pickle.load(file)
            self.centered_frustum_mean_xyz = self.centered_frustum_mean_xyz.astype(np.float32)

        img_ids = []
        img_names = os.listdir(self.img_dir)
        for img_name in img_names:
            img_id = img_name.split(".png")[0]
            img_ids.append(img_id)

        self.examples = []
        for img_id in img_ids:
            detections_file_path = self.detections_2d_dir + img_id + ".txt"
            with open(detections_file_path) as file:
                # line format: img_id, img_height, img_width, class, u_min, v_min, u_max, v_max, confidence score, distance estimate
                for line in file:
                    values = line.split()
                    object_class = float(values[3])
                    if object_class == 1: # (1: Car)
                        u_min = float(values[4])
                        v_min = float(values[5])
                        u_max = float(values[6])
                        v_max = float(values[7])
                        score_2d = float(values[8])

                        detection_2d = {}
                        detection_2d["u_min"] = u_min
                        detection_2d["v_min"] = v_min
                        detection_2d["u_max"] = u_max
                        detection_2d["v_max"] = v_max
                        detection_2d["score_2d"] = score_2d
                        detection_2d["img_id"] = img_id

                        self.examples.append(detection_2d)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        lidar_path = self.lidar_dir + img_id + ".bin"
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        orig_point_cloud = point_cloud

        # remove points that are located behind the camera:
        point_cloud = point_cloud[point_cloud[:, 0] > -5, :]
        # remove points that are located too far away from the camera:
        point_cloud = point_cloud[point_cloud[:, 0] < 80, :]

        calib = calibread(self.calib_dir + img_id + ".txt")
        P2 = calib["P2"]
        Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
        R0_rect_orig = calib["R0_rect"]
        #
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        #
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

        # project the points onto the image plane (homogeneous coords):
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]

        # transform the points into (rectified) camera coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera

        ########################################################################
        # frustum:
        ########################################################################
        u_min = example["u_min"] # (left)
        u_max = example["u_max"] # (rigth)
        v_min = example["v_min"] # (top)
        v_max = example["v_max"] # (bottom)

        score_2d = example["score_2d"]

        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])

        row_mask = np.logical_and(
                    np.logical_and(img_points[:, 0] >= u_min,
                                   img_points[:, 0] <= u_max),
                    np.logical_and(img_points[:, 1] >= v_min,
                                   img_points[:, 1] <= v_max))

        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :] # (needed only for visualization)
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]

        empty_frustum_flag = 0
        if frustum_point_cloud.shape[0] == 0:
            empty_frustum_flag = 1
            frustum_point_cloud = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz = np.zeros((1024, 3), dtype=np.float32)
            frustum_point_cloud_camera = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz_camera = np.zeros((1024, 3), dtype=np.float32)

        # randomly sample 1024 points in the frustum point cloud:
        if frustum_point_cloud.shape[0] < 1024:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=True)
        else:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=False)
        frustum_point_cloud_xyz = frustum_point_cloud_xyz[row_idx, :]
        frustum_point_cloud = frustum_point_cloud[row_idx, :]
        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        frustum_point_cloud_camera = frustum_point_cloud_camera[row_idx, :]
        # (the frustum point cloud now has exactly 1024 points)

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        ########################################################################
        # normalize frustum point cloud:
        ########################################################################
        # get the 2dbbox center img point in hom. coords:
        u_center = u_min + (u_max - u_min)/2.0
        v_center = v_min + (v_max - v_min)/2.0
        center_img_point_hom = np.array([u_center, v_center, 1])

        # (more than one 3D point is projected onto the center image point, i.e,
        # the linear system of equations is under-determined and has inf number
        # of solutions. By using the pseudo-inverse, we obtain the least-norm sol)

        # get a point (the least-norm sol.) that projects onto the center image point, in hom. coords:
        P2_pseudo_inverse = np.linalg.pinv(P2) # (shape: (4, 3)) (P2 has shape (3, 4))
        point_hom = np.dot(P2_pseudo_inverse, center_img_point_hom)

        # hom --> normal coords:
        point = np.array(([point_hom[0]/point_hom[3], point_hom[1]/point_hom[3], point_hom[2]/point_hom[3]]))

        # if the point is behind the camera, switch to the mirror point in front of the camera:
        if point[2] < 0:
            point[0] = -point[0]
            point[2] = -point[2]

        # compute the angle of the point in the x-z plane: ((rectified) camera coords)
        frustum_angle = np.arctan2(point[0], point[2]) # (np.arctan2(x, z)) # (frustum_angle = 0: frustum is centered)

        # rotation_matrix to rotate points frustum_angle around the y axis (counter-clockwise):
        frustum_R = np.asarray([[np.cos(frustum_angle), 0, -np.sin(frustum_angle)],
                           [0, 1, 0],
                           [np.sin(frustum_angle), 0, np.cos(frustum_angle)]],
                           dtype='float32')

        # rotate the frustum point cloud to center it:
        centered_frustum_point_cloud_xyz_camera = np.dot(frustum_R, frustum_point_cloud_xyz_camera.T).T

        # subtract the centered frustum train xyz mean:
        centered_frustum_point_cloud_xyz_camera -= self.centered_frustum_mean_xyz

        centered_frustum_point_cloud_camera = frustum_point_cloud_camera
        centered_frustum_point_cloud_camera[:, 0:3] = centered_frustum_point_cloud_xyz_camera

        # # # # # # # # # # debug visualizations START:
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        # frustum_pcd_camera = PointCloud()
        # frustum_pcd_camera.points = Vector3dVector(frustum_point_cloud_xyz_camera)
        # frustum_pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        # centered_frustum_pcd_camera = PointCloud()
        # centered_frustum_pcd_camera.points = Vector3dVector(centered_frustum_point_cloud_xyz_camera)
        # centered_frustum_pcd_camera.paint_uniform_color([1, 0, 0])
        # draw_geometries([centered_frustum_pcd_camera, frustum_pcd_camera])
        # # # # # # # # # # debug visualizations START:

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)

        centered_frustum_point_cloud_camera = torch.from_numpy(centered_frustum_point_cloud_camera) # (shape: (1024, 3))
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))

        return (centered_frustum_point_cloud_camera, bbox_2d_img, img_id, input_2Dbbox, frustum_R, frustum_angle, empty_frustum_flag, self.centered_frustum_mean_xyz, self.mean_car_size, score_2d)

    def __len__(self):
        return self.num_examples

class DatasetKittiVal2ddetections(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, NH):
        self.img_dir = kitti_data_path + "/object/training/image_2/"
        self.calib_dir = kitti_data_path + "/object/training/calib/"
        self.lidar_dir = kitti_data_path + "/object/training/velodyne/"
        self.detections_2d_path = kitti_meta_path + "/rgb_detection_val.txt"

        self.NH = NH

        with open(kitti_meta_path + "/val_img_ids.pkl", "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)

        with open(kitti_meta_path + "/kitti_centered_frustum_mean_xyz.pkl", "rb") as file: # (needed for python3)
            self.centered_frustum_mean_xyz = pickle.load(file)
            self.centered_frustum_mean_xyz = self.centered_frustum_mean_xyz.astype(np.float32)

        self.examples = []
        with open(self.detections_2d_path) as file:
            # line format: /home/rqi/Data/KITTI/object/training/image_2/***img_id***.png, class, conf_score, u_min, v_min, u_max, v_max
            for line in file:
                values = line.split()
                object_class = float(values[1])
                if object_class == 2: # (2: Car)
                    score_2d = float(values[2])
                    u_min = float(values[3])
                    v_min = float(values[4])
                    u_max = float(values[5])
                    v_max = float(values[6])

                    img_id = values[0].split("image_2/")[1]
                    img_id = img_id.split(".")[0]

                    detection_2d = {}
                    detection_2d["u_min"] = u_min
                    detection_2d["v_min"] = v_min
                    detection_2d["u_max"] = u_max
                    detection_2d["v_max"] = v_max
                    detection_2d["score_2d"] = score_2d
                    detection_2d["img_id"] = img_id

                    self.examples.append(detection_2d)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        lidar_path = self.lidar_dir + img_id + ".bin"
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        orig_point_cloud = point_cloud

        # remove points that are located behind the camera:
        point_cloud = point_cloud[point_cloud[:, 0] > -5, :]
        # remove points that are located too far away from the camera:
        point_cloud = point_cloud[point_cloud[:, 0] < 80, :]

        calib = calibread(self.calib_dir + img_id + ".txt")
        P2 = calib["P2"]
        Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
        R0_rect_orig = calib["R0_rect"]
        #
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        #
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

        # project the points onto the image plane (homogeneous coords):
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]

        # transform the points into (rectified) camera coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera

        ########################################################################
        # frustum:
        ########################################################################
        u_min = example["u_min"] # (left)
        u_max = example["u_max"] # (rigth)
        v_min = example["v_min"] # (top)
        v_max = example["v_max"] # (bottom)

        score_2d = example["score_2d"]

        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])

        row_mask = np.logical_and(
                    np.logical_and(img_points[:, 0] >= u_min,
                                   img_points[:, 0] <= u_max),
                    np.logical_and(img_points[:, 1] >= v_min,
                                   img_points[:, 1] <= v_max))

        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :] # (needed only for visualization)
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]

        empty_frustum_flag = 0
        if frustum_point_cloud.shape[0] == 0:
            empty_frustum_flag = 1
            frustum_point_cloud = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz = np.zeros((1024, 3), dtype=np.float32)
            frustum_point_cloud_camera = np.zeros((1024, 4), dtype=np.float32)
            frustum_point_cloud_xyz_camera = np.zeros((1024, 3), dtype=np.float32)

        # randomly sample 1024 points in the frustum point cloud:
        if frustum_point_cloud.shape[0] < 1024:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=True)
        else:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=False)
        frustum_point_cloud_xyz = frustum_point_cloud_xyz[row_idx, :]
        frustum_point_cloud = frustum_point_cloud[row_idx, :]
        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        frustum_point_cloud_camera = frustum_point_cloud_camera[row_idx, :]
        # (the frustum point cloud now has exactly 1024 points)

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        ########################################################################
        # normalize frustum point cloud:
        ########################################################################
        # get the 2dbbox center img point in hom. coords:
        u_center = u_min + (u_max - u_min)/2.0
        v_center = v_min + (v_max - v_min)/2.0
        center_img_point_hom = np.array([u_center, v_center, 1])

        # (more than one 3D point is projected onto the center image point, i.e,
        # the linear system of equations is under-determined and has inf number
        # of solutions. By using the pseudo-inverse, we obtain the least-norm sol)

        # get a point (the least-norm sol.) that projects onto the center image point, in hom. coords:
        P2_pseudo_inverse = np.linalg.pinv(P2) # (shape: (4, 3)) (P2 has shape (3, 4))
        point_hom = np.dot(P2_pseudo_inverse, center_img_point_hom)

        # hom --> normal coords:
        point = np.array(([point_hom[0]/point_hom[3], point_hom[1]/point_hom[3], point_hom[2]/point_hom[3]]))

        # if the point is behind the camera, switch to the mirror point in front of the camera:
        if point[2] < 0:
            point[0] = -point[0]
            point[2] = -point[2]

        # compute the angle of the point in the x-z plane: ((rectified) camera coords)
        frustum_angle = np.arctan2(point[0], point[2]) # (np.arctan2(x, z)) # (frustum_angle = 0: frustum is centered)

        # rotation_matrix to rotate points frustum_angle around the y axis (counter-clockwise):
        frustum_R = np.asarray([[np.cos(frustum_angle), 0, -np.sin(frustum_angle)],
                           [0, 1, 0],
                           [np.sin(frustum_angle), 0, np.cos(frustum_angle)]],
                           dtype='float32')

        # rotate the frustum point cloud to center it:
        centered_frustum_point_cloud_xyz_camera = np.dot(frustum_R, frustum_point_cloud_xyz_camera.T).T

        # subtract the centered frustum train xyz mean:
        centered_frustum_point_cloud_xyz_camera -= self.centered_frustum_mean_xyz

        centered_frustum_point_cloud_camera = frustum_point_cloud_camera
        centered_frustum_point_cloud_camera[:, 0:3] = centered_frustum_point_cloud_xyz_camera

        # # # # # # # # # # debug visualizations START:
        # import sys
        # sys.path.append("/home/fregu856/exjobb/Open3D/build/lib")
        # from py3d import *
        # frustum_pcd_camera = PointCloud()
        # frustum_pcd_camera.points = Vector3dVector(frustum_point_cloud_xyz_camera)
        # frustum_pcd_camera.paint_uniform_color([0.25, 0.25, 0.25])
        # centered_frustum_pcd_camera = PointCloud()
        # centered_frustum_pcd_camera.points = Vector3dVector(centered_frustum_point_cloud_xyz_camera)
        # centered_frustum_pcd_camera.paint_uniform_color([1, 0, 0])
        # draw_geometries([centered_frustum_pcd_camera, frustum_pcd_camera])
        # # # # # # # # # # debug visualizations END:

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)

        centered_frustum_point_cloud_camera = torch.from_numpy(centered_frustum_point_cloud_camera) # (shape: (1024, 3))
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))

        return (centered_frustum_point_cloud_camera, bbox_2d_img, img_id, input_2Dbbox, frustum_R, frustum_angle, empty_frustum_flag, self.centered_frustum_mean_xyz, self.mean_car_size, score_2d)

    def __len__(self):
        return self.num_examples
