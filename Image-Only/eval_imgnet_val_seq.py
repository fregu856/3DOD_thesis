# camera-ready

from datasets_imgnet import DatasetImgNetEvalValSeq, BoxRegressor # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
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

sequence = "0004"

batch_size = 8

network = ImgNet("Image-Only_eval_val_seq", project_dir="/root/3DOD_thesis")
network.load_state_dict(torch.load("/root/3DOD_thesis/pretrained_models/model_10_2_epoch_400.pth"))
network = network.cuda()

val_dataset = DatasetImgNetEvalValSeq(kitti_data_path="/root/3DOD_thesis/data/kitti",
                                     kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
                                     sequence=sequence)

num_val_batches = int(len(val_dataset)/batch_size)

print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=4)

regression_loss_func = nn.SmoothL1Loss()

network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
batch_losses = []
batch_losses_size = []
batch_losses_keypoints = []
batch_losses_distance = []
batch_losses_3d_center = []
batch_losses_3d_size = []
batch_losses_3d_r_y = []
batch_losses_3d_distance = []
eval_dict = {}
for step, (bbox_2d_imgs, labels_size, labels_keypoints, labels_distance, img_ids, mean_car_size, ws, hs, u_centers, v_centers, camera_matrices, labels_center, labels_r_y, mean_distance) in enumerate(val_loader):
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        print ("step %d/%d" % (step+1, num_val_batches))

        bbox_2d_imgs = Variable(bbox_2d_imgs) # (shape: (batch_size, 3, H, W) = (batch_size, 3, 224, 224))
        labels_size = Variable(labels_size) # (shape: (batch_size, 3))
        labels_keypoints = Variable(labels_keypoints) # (shape: (batch_size, 2*8))
        labels_distance = Variable(labels_distance) # (shape: (batch_size, 1))
        labels_distance = labels_distance.view(-1) # (shape: (batch_size, ))

        bbox_2d_imgs = bbox_2d_imgs.cuda()
        labels_size = labels_size.cuda()
        labels_keypoints = labels_keypoints.cuda()
        labels_distance = labels_distance.cuda()

        outputs = network(bbox_2d_imgs) # (shape: (batch_size, 2*8 + 3 = 19))
        outputs_keypoints = outputs[:, 0:16] # (shape: (batch_size, 2*8))
        outputs_size = outputs[:, 16:19] # (shape: (batch_size, 3)
        outputs_distance = outputs[:, 19] # (shape: (batch_size, )

        ########################################################################
        # compute the size loss:
        ########################################################################
        loss_size = regression_loss_func(outputs_size, labels_size)

        loss_size_value = loss_size.data.cpu().numpy()
        batch_losses_size.append(loss_size_value)

        ########################################################################
        # compute the keypoints loss:
        ########################################################################
        loss_keypoints = regression_loss_func(outputs_keypoints, labels_keypoints)

        loss_keypoints_value = loss_keypoints.data.cpu().numpy()
        batch_losses_keypoints.append(loss_keypoints_value)

        ########################################################################
        # compute the distance loss:
        ########################################################################
        loss_distance = regression_loss_func(outputs_distance, labels_distance)

        loss_distance_value = loss_distance.data.cpu().numpy()
        batch_losses_distance.append(loss_distance_value)

        ########################################################################
        # compute the total loss:
        ########################################################################
        loss = loss_size + 10*loss_keypoints + 0.01*loss_distance
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        ########################################################################
        mean_car_size = Variable(mean_car_size).cuda()
        mean_distance = Variable(mean_distance).cuda()
        outputs_size = outputs_size + mean_car_size # NOTE NOTE
        labels_size = labels_size + mean_car_size # NOTE NOTE
        outputs_distance = outputs_distance + mean_distance # NOTE NOTE
        preds_3d_size = torch.zeros((outputs.size()[0], 3))
        preds_3d_center = torch.zeros((outputs.size()[0], 3))
        preds_3d_r_y = torch.zeros((outputs.size()[0], ))
        preds_3d_distance = torch.zeros((outputs.size()[0], ))
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
            gt_center = labels_center[i]
            gt_size = labels_size[i]
            gt_r_y = labels_r_y[i]

            #if img_id in ["000006", "000007", "000008", "000009", "000010", "000011", "000012", "000013", "000014", "000015", "000016", "000017", "000018", "000019", "000020", "000021"]:
            output_keypoints = np.resize(output_keypoints, (8, 2))
            output_keypoints = output_keypoints*np.array([w, h]) + np.array([u_center, v_center])

            box_regressor = BoxRegressor(camera_matrix=camera_matrix,
                                         pred_size=output_size,
                                         pred_keypoints=output_keypoints,
                                         pred_distance=output_distance)

            pred_params = box_regressor.solve()
            pred_h, pred_w, pred_l, pred_x, pred_y, pred_z, pred_r_y  = pred_params
            pred_r_y = wrapToPi(pred_r_y)

            preds_3d_size[i, 0] = pred_h
            preds_3d_size[i, 1] = pred_w
            preds_3d_size[i, 2] = pred_l

            preds_3d_center[i, 0] = pred_x
            preds_3d_center[i, 1] = pred_y
            preds_3d_center[i, 2] = pred_z

            preds_3d_r_y[i] = pred_r_y

            preds_3d_distance[i] = np.linalg.norm(np.array([pred_x, pred_y, pred_z]))

            gt_r_y = gt_r_y.data.cpu().numpy()

            if img_id not in eval_dict:
                eval_dict[img_id] = []

            bbox_dict = {}
            bbox_dict["pred_center_BboxNet"] = np.array([pred_x, pred_y, pred_z])
            bbox_dict["pred_h"] = pred_h
            bbox_dict["pred_w"] = pred_w
            bbox_dict["pred_l"] = pred_l
            bbox_dict["pred_r_y"] = pred_r_y
            bbox_dict["gt_center"] = gt_center.numpy()
            bbox_dict["gt_h"] = gt_size[0].data.cpu().numpy()
            bbox_dict["gt_w"] = gt_size[1].data.cpu().numpy()
            bbox_dict["gt_l"] = gt_size[2].data.cpu().numpy()
            bbox_dict["gt_r_y"] = gt_r_y

            eval_dict[img_id].append(bbox_dict)

        preds_3d_size = Variable(preds_3d_size).cuda()
        preds_3d_size = preds_3d_size - mean_car_size # NOTE NOTE

        labels_size = labels_size - mean_car_size # NOTE NOTE

        preds_3d_distance = Variable(preds_3d_distance).cuda()
        preds_3d_distance = preds_3d_distance - mean_distance # NOTE NOTE

        preds_3d_center = Variable(preds_3d_center).cuda()
        preds_3d_r_y = Variable(preds_3d_r_y).cuda()

        labels_center = Variable(labels_center).cuda()
        labels_r_y = Variable(labels_r_y).cuda()

        loss_3d_size = regression_loss_func(preds_3d_size, labels_size)
        loss_3d_size_value = loss_3d_size.data.cpu().numpy()
        batch_losses_3d_size.append(loss_3d_size_value)

        loss_3d_center = regression_loss_func(preds_3d_center, labels_center)
        loss_3d_center_value = loss_3d_center.data.cpu().numpy()
        batch_losses_3d_center.append(loss_3d_center_value)

        loss_3d_r_y = regression_loss_func(preds_3d_r_y, labels_r_y)
        loss_3d_r_y_value = loss_3d_r_y.data.cpu().numpy()
        batch_losses_3d_r_y.append(loss_3d_r_y_value)

        loss_3d_distance = regression_loss_func(preds_3d_distance, labels_distance)
        loss_3d_distance_value = loss_3d_distance.data.cpu().numpy()
        batch_losses_3d_distance.append(loss_3d_distance_value)

epoch_loss = np.mean(batch_losses)
print ("val loss: %g" % epoch_loss)

epoch_loss = np.mean(batch_losses_size)
print ("val size loss: %g" % epoch_loss)

epoch_loss = np.mean(batch_losses_keypoints)
print ("val keypoints loss: %g" % epoch_loss)

epoch_loss = np.mean(batch_losses_distance)
print ("val distance loss: %g" % epoch_loss)

epoch_loss = np.mean(batch_losses_3d_size)
print ("val 3d size loss: %g" % epoch_loss)

epoch_loss = np.mean(batch_losses_3d_center)
print ("val 3d center loss: %g" % epoch_loss)

epoch_loss = np.mean(batch_losses_3d_r_y)
print ("val 3d r_y loss: %g" % epoch_loss)

epoch_loss = np.mean(batch_losses_3d_distance)
print ("val 3d distance loss: %g" % epoch_loss)

with open("%s/eval_dict_val_seq_%s.pkl" % (network.model_dir, sequence), "wb") as file:
    pickle.dump(eval_dict, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
