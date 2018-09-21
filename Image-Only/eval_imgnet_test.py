# camera-ready

from datasets_imgnet import DatasetKittiTest, BoxRegressor # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from datasets_imgnet import wrapToPi
from imgnet import ImgNet

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

batch_size = 8

network = ImgNet("Image-Only_eval_test", project_dir="/root/3DOD_thesis")
network.load_state_dict(torch.load("/root/3DOD_thesis/pretrained_models/model_10_2_epoch_400.pth"))
network = network.cuda()

network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)

val_dataset = DatasetKittiTest(kitti_data_path="/root/3DOD_thesis/data/kitti",
                               kitti_meta_path="/root/3DOD_thesis/data/kitti/meta")

num_val_batches = int(len(val_dataset)/batch_size)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=4)

eval_dict = {}
for step, (bbox_2d_imgs, img_ids, mean_car_size, ws, hs, u_centers, v_centers, input_2Dbboxes, camera_matrices, mean_distance, scores_2d) in enumerate(val_loader):
    if step % 100 == 0:
        print ("step: %d/%d" % (step+1, num_val_batches))

    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        bbox_2d_imgs = Variable(bbox_2d_imgs) # (shape: (batch_size, 3, H, W) = (batch_size, 3, 224, 224))

        bbox_2d_imgs = bbox_2d_imgs.cuda()

        outputs = network(bbox_2d_imgs) # (shape: (batch_size, 2*8 + 3 = 19))
        outputs_keypoints = outputs[:, 0:16] # (shape: (batch_size, 2*8))
        outputs_size = outputs[:, 16:19] # (shape: (batch_size, 3)
        outputs_distance = outputs[:, 19] # (shape: (batch_size, )

        ############################################################################
        # save data for visualization:
        ############################################################################
        mean_car_size = Variable(mean_car_size).cuda()
        outputs_size = outputs_size + mean_car_size # NOTE NOTE
        mean_distance = Variable(mean_distance).cuda()
        outputs_distance = outputs_distance + mean_distance # NOTE NOTE
        for i in range(outputs.size()[0]):
            output_keypoints = outputs_keypoints[i].data.cpu().numpy()
            output_size = outputs_size[i].data.cpu().numpy()
            output_distance = outputs_distance[i].data.cpu().numpy()
            w = ws[i]
            h = hs[i]
            u_center = u_centers[i]
            v_center = v_centers[i]
            img_id = img_ids[i]
            camera_matrix = camera_matrices[i]
            input_2Dbbox = input_2Dbboxes[i]
            score_2d = scores_2d[i]

            output_keypoints = np.resize(output_keypoints, (8, 2))
            output_keypoints = output_keypoints*np.array([w, h]) + np.array([u_center, v_center])

            box_regressor = BoxRegressor(camera_matrix=camera_matrix,
                                         pred_size=output_size,
                                         pred_keypoints=output_keypoints,
                                         pred_distance=output_distance)

            pred_params = box_regressor.solve()
            pred_h, pred_w, pred_l, pred_x, pred_y, pred_z, pred_r_y  = pred_params
            pred_r_y = wrapToPi(pred_r_y)

            score_2d = score_2d.data.cpu().numpy()
            input_2Dbbox = input_2Dbbox.data.cpu().numpy()

            if img_id not in eval_dict:
                eval_dict[img_id] = []

            bbox_dict = {}
            bbox_dict["pred_center_BboxNet"] = np.array([pred_x, pred_y, pred_z])
            bbox_dict["pred_h"] = pred_h
            bbox_dict["pred_w"] = pred_w
            bbox_dict["pred_l"] = pred_l
            bbox_dict["pred_r_y"] = pred_r_y
            bbox_dict["input_2Dbbox"] = input_2Dbbox
            bbox_dict["score_2d"] = score_2d

            eval_dict[img_id].append(bbox_dict)

with open("%s/eval_dict_test.pkl" % network.model_dir, "wb") as file:
    pickle.dump(eval_dict, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
