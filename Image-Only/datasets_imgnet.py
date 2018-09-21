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
from scipy.optimize import least_squares

def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def get_keypoints(center, h, w, l, r_y, P2_mat):
    Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)],
                       [0, 1, 0],
                       [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype='float32')

    # get the keypoints in 3d camera coords:
    p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())
    keypoints_3d = np.array([p0, p1, p2, p3, p4, p5, p6, p7]) # (shape: (8, 3))

    # convert to homogeneous coords:
    keypoints_3d_hom = np.ones((8, 4), dtype=np.float32) # (shape: (8, 4))
    keypoints_3d_hom[:, 0:3] = keypoints_3d

    # project onto the image plane:
    keypoints_hom = np.dot(P2_mat, keypoints_3d_hom.T).T # (shape: (8, 3))
    # normalize:
    keypoints = np.zeros((8, 2), dtype=np.float32)
    keypoints[:, 0] = keypoints_hom[:, 0]/keypoints_hom[:, 2]
    keypoints[:, 1] = keypoints_hom[:, 1]/keypoints_hom[:, 2]

    return keypoints # (shape: (8, 2))

def draw_3dbbox_from_keypoints(img, keypoints):
    img = np.copy(img)

    color = [190, 0, 255] # (BGR)
    front_color = [255, 230, 0] # (BGR)
    lines = [[0, 3, 7, 4, 0], [1, 2, 6, 5, 1], [0, 1], [2, 3], [6, 7], [4, 5]] # (0 -> 3 -> 7 -> 4 -> 0, 1 -> 2 -> 6 -> 5 -> 1, etc.)
    colors = [front_color, color, color, color, color, color]

    for n, line in enumerate(lines):
        bg = colors[n]

        cv2.polylines(img, np.int32([keypoints[line]]), False, bg, lineType=cv2.LINE_AA, thickness=2)

    return img

class BoxRegressor(object):
    # NOTE! based on code provided by Eskil Jörgensen

    def __init__(self, camera_matrix, pred_size, pred_keypoints, pred_distance):
        super(BoxRegressor, self).__init__()

        self.P = camera_matrix
        self.P_pseudo_inverse = np.linalg.pinv(self.P)
        self.pred_keypoints = pred_keypoints # (shape: (8, 2))
        self.pred_size = pred_size
        self.pred_distance = pred_distance

    def _residuals(self, params):
        [h, w, l, x, y, z, rot_y] = params

        projected_keypoints = get_keypoints(np.array([x, y, z]), h, w, l, rot_y, self.P) # (shape: (8, 2))

        resids_keypoints = projected_keypoints - self.pred_keypoints # (shape: (8, 2))
        resids_keypoints = resids_keypoints.flatten() # (shape: (16 ,))

        resids_size_regularization = np.array([h - self.pred_size[0],
                                               w - self.pred_size[1],
                                               l - self.pred_size[2]]) # (shape: (3, ))

        resids_distance_regularization = np.array([np.linalg.norm(params[3:6]) - self.pred_distance]) # (shape: (1, ))

        resids = np.append(resids_keypoints, 100*resids_size_regularization) # (shape: (19, ))
        resids = np.append(resids, 10*resids_distance_regularization) # (shape: (20, ))

        return resids

    def _initial_guess(self):
        h, w, l = self.pred_size

        img_keypoints_center_hom = [np.mean(self.pred_keypoints[:, 0]), np.mean(self.pred_keypoints[:, 1]), 1]
        l0 = np.dot(self.P_pseudo_inverse, img_keypoints_center_hom)
        l0 = l0[:3]/l0[3]

        # if the point is behind the camera, switch to the mirror point in front of the camera:
        if l0[2] < 0:
            l0[0] = -l0[0]
            l0[2] = -l0[2]

        [x0, y0, z0] = (l0/np.linalg.norm(l0))*self.pred_distance

        rot_y = -np.pi/2

        return [h, w, l, x0, y0, z0, rot_y]

    def solve(self):
        x0 = self._initial_guess()

        ls_results = []
        costs = []
        for rot_y in [-2, -1, 0, 1]:
            x0[6] = rot_y*np.pi/2

            ls_result = least_squares(self._residuals, x0, jac="3-point")
            ls_results.append(ls_result)
            costs.append(ls_result.cost)

        self.result = ls_results[np.argmin(costs)]
        params = self.result.x

        return params

class DatasetImgNetAugmentation(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, type):
        self.img_dir = kitti_data_path + "/object/training/image_2/"
        self.label_dir = kitti_data_path + "/object/training/label_2/"
        self.calib_dir = kitti_data_path + "/object/training/calib/"
        self.lidar_dir = kitti_data_path + "/object/training/velodyne/"

        with open(kitti_meta_path + "/%s_img_ids_random.pkl" % type, "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)
            self.mean_car_size = self.mean_car_size.astype(np.float32)

        with open(kitti_meta_path + "/kitti_train_mean_distance.pkl", "rb") as file: # (needed for python3)
            self.mean_distance = pickle.load(file)
            self.mean_distance = self.mean_distance.astype(np.float32)
            self.mean_distance = self.mean_distance[0]

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

        label_2D = example["label_2D"]
        label_3D = example["label_3D"]

        bbox = label_2D["poly"]

        u_min = bbox[0, 0] # (left)
        u_max = bbox[1, 0] # (rigth)
        v_min = bbox[0, 1] # (top)
        v_max = bbox[2, 1] # (bottom)

        ########################################################################
        # augment the 2Dbbox:
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

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        # # # # # debug visualization:
        # cv2.imshow("test", bbox_2d_img)
        # cv2.waitKey(0)
        # # # # #

        ########################################################################
        # flip the 2dbbox img crop with 0.5 probability:
        ########################################################################
        # get 0 or 1 with equal probability (indicating if the frustum should be flipped or not):
        flip = np.random.randint(low=0, high=2)

        if flip == 1:
            bbox_2d_img = cv2.flip(bbox_2d_img, 1)
            # cv2.imshow("test", bbox_2d_img)
            # cv2.waitKey(0)

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)

        ########################################################################
        # size ground truth:
        ########################################################################
        label_size = np.zeros((3, ), dtype=np.float32) # ([h, w, l])
        label_size[0] = label_3D['h']
        label_size[1] = label_3D['w']
        label_size[2] = label_3D['l']
        label_size = label_size - self.mean_car_size

        ########################################################################
        # keypoints ground truth:
        ########################################################################
        label_keypoints = get_keypoints(label_3D["center"], label_3D['h'],
                                        label_3D['w'], label_3D['l'],
                                        label_3D['r_y'], label_3D['P0_mat']) # (shape (8, 2))

        if flip == 1:
            img = cv2.imread(self.img_dir + img_id + ".png", -1)
            img_w = img.shape[1]

            u_center = img_w - u_center
            label_keypoints[:, 0] = img_w - label_keypoints[:, 0]

            # swap the keypoints to adjust for the flipping:
            # # swap keypoint 7 and 4:
            temp = np.copy(label_keypoints[7, :])
            label_keypoints[7, :]= label_keypoints[4, :]
            label_keypoints[4, :] = temp
            # # swap keypoint 3 and 0:
            temp = np.copy(label_keypoints[3, :])
            label_keypoints[3, :]= label_keypoints[0, :]
            label_keypoints[0, :] = temp
            # # swap keypoint 6 and 5:
            temp = np.copy(label_keypoints[6, :])
            label_keypoints[6, :]= label_keypoints[5, :]
            label_keypoints[5, :] = temp
            # # swap keypoint 2 and 1:
            temp = np.copy(label_keypoints[2, :])
            label_keypoints[2, :]= label_keypoints[1, :]
            label_keypoints[1, :] = temp

        # # # # # # debug visualization:
        # img = cv2.imread(self.img_dir + img_id + ".png", -1)
        # if flip == 1:
        #     img = cv2.flip(img, 1)
        # img_with_gt_3dbbox = draw_3dbbox_from_keypoints(img, label_keypoints)
        # cv2.imshow("test", img_with_gt_3dbbox)
        # cv2.waitKey(0)
        # # # # # #

        label_keypoints = label_keypoints - np.array([u_center, v_center])
        label_keypoints = label_keypoints/np.array([w, h])

        # # # # # # debug visualization:
        # img = cv2.imread(self.img_dir + img_id + ".png", -1)
        # if flip == 1:
        #     img = cv2.flip(img, 1)
        # keypoints_restored = label_keypoints*np.array([w, h]) + np.array([u_center, v_center])
        # img_with_gt_3dbbox = draw_3dbbox_from_keypoints(img, keypoints_restored)
        # cv2.imshow("test", img_with_gt_3dbbox)
        # cv2.waitKey(0)
        # # # # # #

        label_keypoints = label_keypoints.flatten() # (shape: (2*8 = 16, )) (np.resize(label_keypoints, (8, 2)) to restore)
        label_keypoints = label_keypoints.astype(np.float32)

        ########################################################################
        # distance ground truth:
        ########################################################################
        label_distance = np.array([np.linalg.norm(label_3D["center"])], dtype=np.float32)

        label_distance = label_distance - self.mean_distance

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))
        label_size = torch.from_numpy(label_size) # (shape: (3, ))
        label_keypoints = torch.from_numpy(label_keypoints) # (shape: (2*8 = 16, ))
        label_distance = torch.from_numpy(label_distance) # (shape: (1, ))

        return (bbox_2d_img, label_size, label_keypoints, label_distance)

    def __len__(self):
        return self.num_examples

# test = DatasetImgNetAugmentation("/home/fregu856/exjobb/data/kitti", "/home/fregu856/exjobb/data/kitti/meta", type="train")
# for i in range(15):
#     _ = test.__getitem__(i)

class DatasetImgNetEval(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, type):
        self.img_dir = kitti_data_path + "/object/training/image_2/"
        self.label_dir = kitti_data_path + "/object/training/label_2/"
        self.calib_dir = kitti_data_path + "/object/training/calib/"
        self.lidar_dir = kitti_data_path + "/object/training/velodyne/"

        with open(kitti_meta_path + "/%s_img_ids.pkl" % type, "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)
            self.mean_car_size = self.mean_car_size.astype(np.float32)

        with open(kitti_meta_path + "/kitti_train_mean_distance.pkl", "rb") as file: # (needed for python3)
            self.mean_distance = pickle.load(file)
            self.mean_distance = self.mean_distance.astype(np.float32)
            self.mean_distance = self.mean_distance[0]

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

        label_2D = example["label_2D"]
        label_3D = example["label_3D"]

        bbox = label_2D["poly"]

        u_min = bbox[0, 0] # (left)
        u_max = bbox[1, 0] # (rigth)
        v_min = bbox[0, 1] # (top)
        v_max = bbox[2, 1] # (bottom)

        w = u_max - u_min
        h = v_max - v_min
        u_center = u_min + w/2.0
        v_center = v_min + h/2.0

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        # # # # # debug visualization:
        # cv2.imshow("test", bbox_2d_img)
        # cv2.waitKey(0)
        # # # # #

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)

        ########################################################################
        # size ground truth:
        ########################################################################
        label_size = np.zeros((3, ), dtype=np.float32) # ([h, w, l])
        label_size[0] = label_3D['h']
        label_size[1] = label_3D['w']
        label_size[2] = label_3D['l']
        label_size = label_size - self.mean_car_size

        ########################################################################
        # keypoints ground truth:
        ########################################################################
        label_keypoints = get_keypoints(label_3D["center"], label_3D['h'],
                                        label_3D['w'], label_3D['l'],
                                        label_3D['r_y'], label_3D['P0_mat']) # (shape (8, 2))

        # # # # # # debug visualization:
        # img = cv2.imread(self.img_dir + img_id + ".png", -1)
        # img_with_gt_3dbbox = draw_3dbbox_from_keypoints(img, label_keypoints)
        # cv2.imshow("test", img_with_gt_3dbbox)
        # cv2.waitKey(0)
        # # # # # #

        label_keypoints = label_keypoints - np.array([u_center, v_center])
        label_keypoints = label_keypoints/np.array([w, h])

        # # # # # # debug visualization:
        # img = cv2.imread(self.img_dir + img_id + ".png", -1)
        # keypoints_restored = label_keypoints*np.array([img_w, img_h]) + np.array([u_center, v_center])
        # img_with_gt_3dbbox = draw_3dbbox_from_keypoints(img, keypoints_restored)
        # cv2.imshow("test", img_with_gt_3dbbox)
        # cv2.waitKey(0)
        # # # # # #

        label_keypoints = label_keypoints.flatten() # (shape: (2*8 = 16, )) (np.resize(label_keypoints, (8, 2)) to restore)
        label_keypoints = label_keypoints.astype(np.float32)

        ########################################################################
        # distance ground truth:
        ########################################################################
        label_distance = np.array([np.linalg.norm(label_3D["center"])], dtype=np.float32)

        label_distance = label_distance - self.mean_distance

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))
        label_size = torch.from_numpy(label_size) # (shape: (3, ))
        label_keypoints = torch.from_numpy(label_keypoints) # (shape: (2*8 = 16, ))
        label_distance = torch.from_numpy(label_distance) # (shape: (1, ))

        camera_matrix = label_3D["P0_mat"]
        gt_center = label_3D["center"]
        gt_center = gt_center.astype(np.float32)
        gt_r_y = np.float32(label_3D["r_y"])

        return (bbox_2d_img, label_size, label_keypoints, label_distance, img_id, self.mean_car_size, w, h, u_center, v_center, camera_matrix, gt_center, gt_r_y, self.mean_distance)

    def __len__(self):
        return self.num_examples

# test = DatasetImgNetEval("/home/fregu856/exjobb/data/kitti", "/home/fregu856/exjobb/data/kitti/meta", type="train")
# for i in range(15):
#     _ = test.__getitem__(i)

class DatasetImgNetEvalValSeq(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, sequence="0000"):
        self.img_dir = kitti_data_path + "/tracking/training/image_02/" + sequence + "/"
        self.lidar_dir = kitti_data_path + "/tracking/training/velodyne/" + sequence + "/"
        self.label_path = kitti_data_path + "/tracking/training/label_02/" + sequence + ".txt"
        self.calib_path = kitti_meta_path + "/tracking/training/calib/" + sequence + ".txt" # NOTE! NOTE! the data format for the calib files was sliightly different for tracking, so I manually modifed the 20 files and saved them in the kitti_meta folder

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)
            self.mean_car_size = self.mean_car_size.astype(np.float32)

        with open(kitti_meta_path + "/kitti_train_mean_distance.pkl", "rb") as file: # (needed for python3)
            self.mean_distance = pickle.load(file)
            self.mean_distance = self.mean_distance.astype(np.float32)
            self.mean_distance = self.mean_distance[0]

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

        label_2D = example["label_2D"]
        label_3D = example["label_3D"]

        bbox = label_2D["poly"]

        u_min = bbox[0, 0] # (left)
        u_max = bbox[1, 0] # (rigth)
        v_min = bbox[0, 1] # (top)
        v_max = bbox[2, 1] # (bottom)

        w = u_max - u_min
        h = v_max - v_min
        u_center = u_min + w/2.0
        v_center = v_min + h/2.0

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        # # # # # debug visualization:
        #cv2.imshow("test", bbox_2d_img)
        #cv2.waitKey(0)
        # # # # #

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)

        ########################################################################
        # size ground truth:
        ########################################################################
        label_size = np.zeros((3, ), dtype=np.float32) # ([h, w, l])
        label_size[0] = label_3D['h']
        label_size[1] = label_3D['w']
        label_size[2] = label_3D['l']
        label_size = label_size - self.mean_car_size

        ########################################################################
        # keypoints ground truth:
        ########################################################################
        label_keypoints = get_keypoints(label_3D["center"], label_3D['h'],
                                        label_3D['w'], label_3D['l'],
                                        label_3D['r_y'], label_3D['P0_mat']) # (shape (8, 2))

        # # # # # # debug visualization:
        # img = cv2.imread(self.img_dir + img_id + ".png", -1)
        # img_with_gt_3dbbox = draw_3dbbox_from_keypoints(img, label_keypoints)
        # cv2.imshow("test", img_with_gt_3dbbox)
        # cv2.waitKey(0)
        # # # # # #

        label_keypoints = label_keypoints - np.array([u_center, v_center])
        label_keypoints = label_keypoints/np.array([w, h])

        # # # # # # debug visualization:
        # img = cv2.imread(self.img_dir + img_id + ".png", -1)
        # keypoints_restored = label_keypoints*np.array([img_w, img_h]) + np.array([u_center, v_center])
        # img_with_gt_3dbbox = draw_3dbbox_from_keypoints(img, keypoints_restored)
        # cv2.imshow("test", img_with_gt_3dbbox)
        # cv2.waitKey(0)
        # # # # # #

        label_keypoints = label_keypoints.flatten() # (shape: (2*8 = 16, )) (np.resize(label_keypoints, (8, 2)) to restore)
        label_keypoints = label_keypoints.astype(np.float32)

        ########################################################################
        # distance ground truth:
        ########################################################################
        label_distance = np.array([np.linalg.norm(label_3D["center"])], dtype=np.float32)

        label_distance = label_distance - self.mean_distance

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))
        label_size = torch.from_numpy(label_size) # (shape: (3, ))
        label_keypoints = torch.from_numpy(label_keypoints) # (shape: (2*8 = 16, ))
        label_distance = torch.from_numpy(label_distance) # (shape: (1, ))

        camera_matrix = label_3D["P0_mat"]
        gt_center = label_3D["center"]
        gt_center = gt_center.astype(np.float32)
        gt_r_y = np.float32(label_3D["r_y"])

        return (bbox_2d_img, label_size, label_keypoints, label_distance, img_id, self.mean_car_size, w, h, u_center, v_center, camera_matrix, gt_center, gt_r_y, self.mean_distance)

    def __len__(self):
        return self.num_examples

# test = DatasetImgNetEvalValSeq("/home/fregu856/exjobb/data/kitti", "/home/fregu856/exjobb/data/kitti/meta", train_dataset="kitti", sequence="0004")
# for i in range(15):
#     _ = test.__getitem__(i)

class DatasetImgNetEvalTestSeq(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, sequence="0000"):
        self.img_dir = kitti_data_path + "/tracking/testing/image_02/" + sequence + "/"
        self.lidar_dir = kitti_data_path + "/tracking/testing/velodyne/" + sequence + "/"
        self.calib_path = kitti_meta_path + "/tracking/testing/calib/" + sequence + ".txt" # NOTE! NOTE! the data format for the calib files was sliightly different for tracking, so I manually modifed the 28 files and saved them in the kitti_meta folder
        self.detections_2d_path = kitti_meta_path + "/tracking/testing/2d_detections/" + sequence + "/inferResult_1.txt"

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)
            self.mean_car_size = self.mean_car_size.astype(np.float32)

        with open(kitti_meta_path + "/kitti_train_mean_distance.pkl", "rb") as file: # (needed for python3)
            self.mean_distance = pickle.load(file)
            self.mean_distance = self.mean_distance.astype(np.float32)
            self.mean_distance = self.mean_distance[0]

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

        calib = calibread(self.calib_path)
        camera_matrix = calib['P2']

        u_min = example["u_min"] # (left)
        u_max = example["u_max"] # (rigth)
        v_min = example["v_min"] # (top)
        v_max = example["v_max"] # (bottom)

        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])

        w = u_max - u_min
        h = v_max - v_min
        u_center = u_min + w/2.0
        v_center = v_min + h/2.0

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        # # # # # debug visualization:
        #cv2.imshow("test", bbox_2d_img)
        #cv2.waitKey(0)
        # # # # #

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))

        return (bbox_2d_img, img_id, self.mean_car_size, w, h, u_center, v_center, input_2Dbbox, camera_matrix, self.mean_distance)

    def __len__(self):
        return self.num_examples

class DatasetKittiTest(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path):
        self.img_dir = kitti_data_path + "/object/testing/image_2/"
        self.calib_dir = kitti_data_path + "/object/testing/calib/"
        self.lidar_dir = kitti_data_path + "/object/testing/velodyne/"
        self.detections_2d_dir = kitti_meta_path + "/object/testing/2d_detections/"

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)
            self.mean_car_size = self.mean_car_size.astype(np.float32)

        with open(kitti_meta_path + "/kitti_train_mean_distance.pkl", "rb") as file: # (needed for python3)
            self.mean_distance = pickle.load(file)
            self.mean_distance = self.mean_distance.astype(np.float32)
            self.mean_distance = self.mean_distance[0]

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

        calib_path = self.calib_dir + img_id + ".txt"
        calib = calibread(calib_path)
        camera_matrix = calib['P2']

        u_min = example["u_min"] # (left)
        u_max = example["u_max"] # (rigth)
        v_min = example["v_min"] # (top)
        v_max = example["v_max"] # (bottom)

        score_2d = example["score_2d"]

        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])

        w = u_max - u_min
        h = v_max - v_min
        u_center = u_min + w/2.0
        v_center = v_min + h/2.0

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        # # # # # debug visualization:
        #cv2.imshow("test", bbox_2d_img)
        #cv2.waitKey(0)
        # # # # #

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))

        return (bbox_2d_img, img_id, self.mean_car_size, w, h, u_center, v_center, input_2Dbbox, camera_matrix, self.mean_distance, score_2d)

    def __len__(self):
        return self.num_examples

class DatasetImgNetVal2ddetections(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path):
        self.img_dir = kitti_data_path + "/object/training/image_2/"
        self.calib_dir = kitti_data_path + "/object/training/calib/"
        self.lidar_dir = kitti_data_path + "/object/training/velodyne/"
        self.detections_2d_path = kitti_meta_path + "/rgb_detection_val.txt"

        with open(kitti_meta_path + "/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)
            self.mean_car_size = self.mean_car_size.astype(np.float32)

        with open(kitti_meta_path + "/kitti_train_mean_distance.pkl", "rb") as file: # (needed for python3)
            self.mean_distance = pickle.load(file)
            self.mean_distance = self.mean_distance.astype(np.float32)
            self.mean_distance = self.mean_distance[0]

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

        calib_path = self.calib_dir + img_id + ".txt"
        calib = calibread(calib_path)
        camera_matrix = calib['P2']

        u_min = example["u_min"] # (left)
        u_max = example["u_max"] # (rigth)
        v_min = example["v_min"] # (top)
        v_max = example["v_max"] # (bottom)

        score_2d = example["score_2d"]

        input_2Dbbox = np.array([u_min, u_max, v_min, v_max])

        w = u_max - u_min
        h = v_max - v_min
        u_center = u_min + w/2.0
        v_center = v_min + h/2.0

        ########################################################################
        # get the input 2dbbox img crop and resize to 224 x 224:
        ########################################################################
        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)

        bbox_2d_img = img[int(np.max([0, v_min])):int(v_max), int(np.max([0, u_min])):int(u_max)]
        bbox_2d_img = cv2.resize(bbox_2d_img, (224, 224))

        # # # # # debug visualization:
        #cv2.imshow("test", bbox_2d_img)
        #cv2.waitKey(0)
        # # # # #

        ########################################################################
        # normalize the 2dbbox img crop:
        ########################################################################
        bbox_2d_img = bbox_2d_img/255.0
        bbox_2d_img = bbox_2d_img - np.array([0.485, 0.456, 0.406])
        bbox_2d_img = bbox_2d_img/np.array([0.229, 0.224, 0.225]) # (shape: (H, W, 3))
        bbox_2d_img = np.transpose(bbox_2d_img, (2, 0, 1)) # (shape: (3, H, W))
        bbox_2d_img = bbox_2d_img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        bbox_2d_img = torch.from_numpy(bbox_2d_img) # (shape: (3, H, W) = (3, 224, 224))

        return (bbox_2d_img, img_id, self.mean_car_size, w, h, u_center, v_center, input_2Dbbox, camera_matrix, self.mean_distance, score_2d)

    def __len__(self):
        return self.num_examples

# ###############################################################################
# # compute mean_distance in the KITTI train set:
# ###############################################################################
# from kittiloader import LabelLoader2D3D
#
# import pickle
# import numpy as np
#
# with open("/staging/frexgus/kitti/meta/train_img_ids.pkl", "rb") as file:
#     img_ids = pickle.load(file)
#
# label_dir = "/datasets/kitti/object/training/label_2/"
# calib_dir = "/datasets/kitti/object/training/calib/"
#
# distances = np.array([])
# for img_id in img_ids:
#     labels = LabelLoader2D3D(img_id, label_dir, ".txt", calib_dir, ".txt")
#     for label in labels:
#         label_3d = label["label_3D"]
#         if label_3d["class"] == "Car":
#             distance = np.linalg.norm(label_3d["center"])
#
#             distances = np.append(distances, distance)
#
# print (distances)
# print (distances.shape)
#
# mean_distance = np.mean(distances)
# mean_distance = np.array([mean_distance])
#
# print (mean_distance)
#
# with open("/staging/frexgus/kitti/meta/kitti_train_mean_distance.pkl", "wb") as file:
#    pickle.dump(mean_distance, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2))
