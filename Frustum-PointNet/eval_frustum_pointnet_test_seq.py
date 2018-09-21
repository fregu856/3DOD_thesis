# camera-ready

from datasets import DatasetKittiTestSequence, wrapToPi, getBinCenter # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from frustum_pointnet import FrustumPointNet

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle

batch_size = 32

network = FrustumPointNet("Frustum-PointNet_eval_test_seq", project_dir="/root/3DOD_thesis")
network.load_state_dict(torch.load("/root/3DOD_thesis/pretrained_models/model_37_2_epoch_400.pth"))
network = network.cuda()

network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)

NH = network.BboxNet_network.NH

for sequence in ["0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0027"]:
    print (sequence)

    test_dataset = DatasetKittiTestSequence(kitti_data_path="/root/3DOD_thesis/data/kitti",
                                            kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
                                            NH=NH, sequence=sequence)

    num_test_batches = int(len(test_dataset)/batch_size)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=16)

    eval_dict = {}
    for step, (frustum_point_clouds, img_ids, input_2Dbboxes, frustum_Rs, frustum_angles, empty_frustum_flags, centered_frustum_mean_xyz, mean_car_size) in enumerate(test_loader):
        if step % 100 == 0:
            print ("step: %d/%d" % (step+1, num_test_batches))

        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            frustum_point_clouds = Variable(frustum_point_clouds) # (shape: (batch_size, num_points, 4))
            frustum_point_clouds = frustum_point_clouds.transpose(2, 1) # (shape: (batch_size, 4, num_points))

            frustum_point_clouds = frustum_point_clouds.cuda()

            outputs = network(frustum_point_clouds)
            outputs_InstanceSeg = outputs[0] # (shape: (batch_size, num_points, 2))
            outputs_TNet = outputs[1] # (shape: (batch_size, 3))
            outputs_BboxNet = outputs[2] # (shape: (batch_size, 3 + 3 + 2*NH))
            seg_point_clouds_mean = outputs[3] # (shape: (batch_size, 3))
            dont_care_mask = outputs[4] # (shape: (batch_size, ))

            ############################################################################
            # save data for visualization:
            ############################################################################
            centered_frustum_mean_xyz = centered_frustum_mean_xyz[0].numpy()
            mean_car_size = mean_car_size[0].numpy()
            for i in range(outputs_InstanceSeg.size()[0]):
                dont_care_mask_value = dont_care_mask[i]
                empty_frustum_flag = empty_frustum_flags[i]

                # don't care about predicted 3Dbboxes that corresponds to empty
                # point clouds outputted by InstanceSeg, or empty input frustums:
                if dont_care_mask_value == 1 and empty_frustum_flag == 0:
                    pred_InstanceSeg = outputs_InstanceSeg[i].data.cpu().numpy() # (shape: (num_points, 2))
                    frustum_point_cloud = frustum_point_clouds[i].transpose(1, 0).data.cpu().numpy() # (shape: (num_points, 4))
                    seg_point_cloud_mean = seg_point_clouds_mean[i].data.cpu().numpy() # (shape: (3, ))
                    img_id = img_ids[i]
                    #if img_id in ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009", "000010", "000011", "000012", "000013", "000014", "000015", "000016", "000017", "000018", "000019", "000020", "000021", "000022", "000023", "000024", "000025", "000026", "000027", "000028", "000029", "000030"]:
                    input_2Dbbox = input_2Dbboxes[i] # (shape: (4, ))
                    frustum_R = frustum_Rs[i] # (shape: (3, 3))
                    frustum_angle = frustum_angles[i]

                    unshifted_frustum_point_cloud_xyz = frustum_point_cloud[:, 0:3] + centered_frustum_mean_xyz
                    decentered_frustum_point_cloud_xyz = np.dot(np.linalg.inv(frustum_R), unshifted_frustum_point_cloud_xyz.T).T
                    frustum_point_cloud[:, 0:3] = decentered_frustum_point_cloud_xyz

                    row_mask = pred_InstanceSeg[:, 1] > pred_InstanceSeg[:, 0]
                    pred_seg_point_cloud = frustum_point_cloud[row_mask, :]

                    pred_center_TNet = np.dot(np.linalg.inv(frustum_R), outputs_TNet[i].data.cpu().numpy() + centered_frustum_mean_xyz + seg_point_cloud_mean) # (shape: (3, )) # NOTE!
                    centroid = seg_point_cloud_mean

                    pred_center_BboxNet = np.dot(np.linalg.inv(frustum_R), outputs_BboxNet[i][0:3].data.cpu().numpy() + centered_frustum_mean_xyz + seg_point_cloud_mean + outputs_TNet[i].data.cpu().numpy()) # (shape: (3, )) # NOTE!

                    pred_h = outputs_BboxNet[i][3].data.cpu().numpy() + mean_car_size[0]
                    pred_w = outputs_BboxNet[i][4].data.cpu().numpy() + mean_car_size[1]
                    pred_l = outputs_BboxNet[i][5].data.cpu().numpy() + mean_car_size[2]

                    pred_bin_scores = outputs_BboxNet[i][6:(6+4)].data.cpu().numpy() # (shape (NH=8, ))
                    pred_residuals = outputs_BboxNet[i][(6+4):].data.cpu().numpy() # (shape (NH=8, ))
                    pred_bin_number = np.argmax(pred_bin_scores)
                    pred_bin_center = getBinCenter(pred_bin_number, NH=NH)
                    pred_residual = pred_residuals[pred_bin_number]
                    pred_centered_r_y = pred_bin_center + pred_residual
                    pred_r_y = wrapToPi(pred_centered_r_y + frustum_angle) # NOTE!

                    pred_r_y = pred_r_y.data.cpu().numpy()
                    input_2Dbbox = input_2Dbbox.data.cpu().numpy()

                    if img_id not in eval_dict:
                        eval_dict[img_id] = []

                    bbox_dict = {}
                    # # # # uncomment this if you want to visualize the frustum or the segmentation:
                    # bbox_dict["frustum_point_cloud"] = frustum_point_cloud
                    # bbox_dict["pred_seg_point_cloud"] = pred_seg_point_cloud
                    # # # #
                    bbox_dict["pred_center_TNet"] = pred_center_TNet
                    bbox_dict["pred_center_BboxNet"] = pred_center_BboxNet
                    bbox_dict["centroid"] = centroid
                    bbox_dict["pred_h"] = pred_h
                    bbox_dict["pred_w"] = pred_w
                    bbox_dict["pred_l"] = pred_l
                    bbox_dict["pred_r_y"] = pred_r_y
                    bbox_dict["input_2Dbbox"] = input_2Dbbox

                    eval_dict[img_id].append(bbox_dict)

    with open("%s/eval_dict_test_seq_%s.pkl" % (network.model_dir, sequence), "wb") as file:
        pickle.dump(eval_dict, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
