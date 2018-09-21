# camera-ready

from datasets import EvalDatasetFrustumPointNet, wrapToPi, getBinCenter # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from frustum_pointnet import FrustumPointNet

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

batch_size = 32

network = FrustumPointNet("Frustum-PointNet_eval_val", project_dir="/root/3DOD_thesis")
network.load_state_dict(torch.load("/root/3DOD_thesis/pretrained_models/model_37_2_epoch_400.pth"))
network = network.cuda()

NH = network.BboxNet_network.NH

val_dataset = EvalDatasetFrustumPointNet(kitti_data_path="/root/3DOD_thesis/data/kitti",
                                         kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
                                         type="val", NH=NH)

num_val_batches = int(len(val_dataset)/batch_size)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=16)

regression_loss_func = nn.SmoothL1Loss()

network.eval() # (set in evaluation mode, this affects BatchNorm, dropout etc.)
batch_losses = []
batch_losses_InstanceSeg = []
batch_losses_TNet = []
batch_losses_BboxNet = []
batch_losses_BboxNet_center = []
batch_losses_BboxNet_size = []
batch_losses_BboxNet_heading_regr = []
batch_losses_BboxNet_heading_class = []
batch_losses_BboxNet_heading_class_weighted = []
batch_losses_corner = []
batch_accuracies = []
batch_precisions = []
batch_recalls = []
batch_f1s = []
batch_accuracies_heading_class = []
eval_dict = {}
for step, (frustum_point_clouds, labels_InstanceSeg, labels_TNet, labels_BboxNet, labels_corner, labels_corner_flipped, img_ids, input_2Dbboxes, frustum_Rs, frustum_angles, centered_frustum_mean_xyz) in enumerate(val_loader):
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        frustum_point_clouds = Variable(frustum_point_clouds) # (shape: (batch_size, num_points, 4))
        labels_InstanceSeg = Variable(labels_InstanceSeg) # (shape: (batch_size, num_points))
        labels_TNet = Variable(labels_TNet) # (shape: (batch_size, 3))
        labels_BboxNet = Variable(labels_BboxNet) # (shape:(batch_size, 11))
        labels_corner = Variable(labels_corner) # (shape: (batch_size, 8, 3))
        labels_corner_flipped = Variable(labels_corner_flipped) # (shape: (batch_size, 8, 3))

        frustum_point_clouds = frustum_point_clouds.transpose(2, 1) # (shape: (batch_size, 4, num_points))

        frustum_point_clouds = frustum_point_clouds.cuda()
        labels_InstanceSeg = labels_InstanceSeg.cuda()
        labels_TNet = labels_TNet.cuda()
        labels_BboxNet = labels_BboxNet.cuda()
        labels_corner = labels_corner.cuda()
        labels_corner_flipped = labels_corner_flipped.cuda()

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
        for i in range(outputs_InstanceSeg.size()[0]):
            dont_care_mask_value = dont_care_mask[i]

            # don't care about predicted 3Dbboxes that corresponds to empty point clouds outputted by InstanceSeg:
            if dont_care_mask_value == 1:
                pred_InstanceSeg = outputs_InstanceSeg[i].data.cpu().numpy() # (shape: (num_points, 2))
                frustum_point_cloud = frustum_point_clouds[i].transpose(1, 0).data.cpu().numpy() # (shape: (num_points, 4))
                label_InstanceSeg = labels_InstanceSeg[i].data.cpu().numpy() # (shape: (num_points, ))
                seg_point_cloud_mean = seg_point_clouds_mean[i].data.cpu().numpy() # (shape: (3, ))
                img_id = img_ids[i]
                #if img_id in ["000006", "000007", "000008", "000009", "000010", "000011", "000012", "000013", "000014", "000015", "000016", "000017", "000018", "000019", "000020", "000021"]:
                input_2Dbbox = input_2Dbboxes[i] # (shape: (4, ))
                frustum_R = frustum_Rs[i] # (shape: (3, 3))
                frustum_angle = frustum_angles[i]

                unshifted_frustum_point_cloud_xyz = frustum_point_cloud[:, 0:3] + centered_frustum_mean_xyz
                decentered_frustum_point_cloud_xyz = np.dot(np.linalg.inv(frustum_R), unshifted_frustum_point_cloud_xyz.T).T
                frustum_point_cloud[:, 0:3] = decentered_frustum_point_cloud_xyz

                row_mask = pred_InstanceSeg[:, 1] > pred_InstanceSeg[:, 0]
                pred_seg_point_cloud = frustum_point_cloud[row_mask, :]

                row_mask = label_InstanceSeg == 1
                gt_seg_point_cloud = frustum_point_cloud[row_mask, :]

                pred_center_TNet = np.dot(np.linalg.inv(frustum_R), outputs_TNet[i].data.cpu().numpy() + centered_frustum_mean_xyz + seg_point_cloud_mean) # (shape: (3, )) # NOTE!
                gt_center = np.dot(np.linalg.inv(frustum_R), labels_TNet[i].data.cpu().numpy() + centered_frustum_mean_xyz) # NOTE!
                centroid = seg_point_cloud_mean

                pred_center_BboxNet = np.dot(np.linalg.inv(frustum_R), outputs_BboxNet[i][0:3].data.cpu().numpy() + centered_frustum_mean_xyz + seg_point_cloud_mean + outputs_TNet[i].data.cpu().numpy()) # (shape: (3, )) # NOTE!

                pred_h = outputs_BboxNet[i][3].data.cpu().numpy() + labels_BboxNet[i][8].data.cpu().numpy()
                pred_w = outputs_BboxNet[i][4].data.cpu().numpy() + labels_BboxNet[i][9].data.cpu().numpy()
                pred_l = outputs_BboxNet[i][5].data.cpu().numpy() + labels_BboxNet[i][10].data.cpu().numpy()

                pred_bin_scores = outputs_BboxNet[i][6:(6+4)].data.cpu().numpy() # (shape (NH=8, ))
                pred_residuals = outputs_BboxNet[i][(6+4):].data.cpu().numpy() # (shape (NH=8, ))
                pred_bin_number = np.argmax(pred_bin_scores)
                pred_bin_center = getBinCenter(pred_bin_number, NH=NH)
                pred_residual = pred_residuals[pred_bin_number]
                pred_centered_r_y = pred_bin_center + pred_residual
                pred_r_y = wrapToPi(pred_centered_r_y + frustum_angle) # NOTE!

                gt_h = labels_BboxNet[i][3].data.cpu().numpy()
                gt_w = labels_BboxNet[i][4].data.cpu().numpy()
                gt_l = labels_BboxNet[i][5].data.cpu().numpy()

                gt_bin_number = labels_BboxNet[i][6].data.cpu().numpy()
                gt_bin_center = getBinCenter(gt_bin_number, NH=NH)
                gt_residual = labels_BboxNet[i][7].data.cpu().numpy()
                gt_centered_r_y = gt_bin_center + gt_residual
                gt_r_y = wrapToPi(gt_centered_r_y + frustum_angle) # NOTE!

                pred_r_y = pred_r_y.data.cpu().numpy()
                gt_r_y = gt_r_y.data.cpu().numpy()
                input_2Dbbox = input_2Dbbox.data.cpu().numpy()

                if img_id not in eval_dict:
                    eval_dict[img_id] = []

                bbox_dict = {}
                # # # # uncomment this if you want to visualize the frustum or the segmentation (e.g., if you want to run visualization/visualize_eval_val_extra.py):
                # bbox_dict["frustum_point_cloud"] = frustum_point_cloud
                # bbox_dict["pred_seg_point_cloud"] = pred_seg_point_cloud
                # bbox_dict["gt_seg_point_cloud"] = gt_seg_point_cloud
                # # # #
                bbox_dict["pred_center_TNet"] = pred_center_TNet
                bbox_dict["pred_center_BboxNet"] = pred_center_BboxNet
                bbox_dict["gt_center"] = gt_center
                bbox_dict["centroid"] = centroid
                bbox_dict["pred_h"] = pred_h
                bbox_dict["pred_w"] = pred_w
                bbox_dict["pred_l"] = pred_l
                bbox_dict["pred_r_y"] = pred_r_y
                bbox_dict["gt_h"] = gt_h
                bbox_dict["gt_w"] = gt_w
                bbox_dict["gt_l"] = gt_l
                bbox_dict["gt_r_y"] = gt_r_y
                bbox_dict["input_2Dbbox"] = input_2Dbbox

                eval_dict[img_id].append(bbox_dict)

        ########################################################################
        # compute precision, recall etc. for the InstanceSeg:
        ########################################################################
        preds = outputs_InstanceSeg.data.cpu().numpy() # (shape: (batch_size, num_points, 2))
        preds = np.argmax(preds, 2) # (shape: (batch_size, num_points))

        labels_InstanceSeg_np = labels_InstanceSeg.data.cpu().numpy() # (shape: (batch_size, num_points))

        accuracy = np.count_nonzero(preds == labels_InstanceSeg_np)/(preds.shape[0]*preds.shape[1])
        if np.count_nonzero(preds == 1) > 0:
            precision = np.count_nonzero(np.logical_and(preds == labels_InstanceSeg_np, preds == 1))/np.count_nonzero(preds == 1) # (TP/(TP + FP))
        else:
            precision = -1
        if np.count_nonzero(labels_InstanceSeg_np == 1) > 0:
            recall = np.count_nonzero(np.logical_and(preds == labels_InstanceSeg_np, preds == 1))/np.count_nonzero(labels_InstanceSeg_np == 1) # (TP/(TP + FN))
        else:
            recall = -1
        if recall + precision > 0:
            f1 = 2*recall*precision/(recall + precision)
        else:
            f1 = -1

        batch_accuracies.append(accuracy)
        if precision != -1:
            batch_precisions.append(precision)
        if recall != -1:
            batch_recalls.append(recall)
        if f1 != -1:
            batch_f1s.append(f1)

        ########################################################################
        # compute accuracy for the heading classification:
        ########################################################################
        pred_bin_scores = outputs_BboxNet[:, 6:(6+NH)].data.cpu().numpy() # (shape: (batch_size, NH))
        pred_bin_numbers = np.argmax(pred_bin_scores, 1) # (shape: (batch_size, ))
        gt_bin_numbers = labels_BboxNet[:, 6].data.cpu().numpy() # (shape: (batch_size, ))

        accuracy_heading_class = np.count_nonzero(pred_bin_numbers == gt_bin_numbers)/(pred_bin_numbers.shape[0])
        batch_accuracies_heading_class.append(accuracy_heading_class)

        ########################################################################
        # compute the InstanceSeg loss:
        ########################################################################
        outputs_InstanceSeg = outputs_InstanceSeg.view(-1, 2) # (shape (batch_size*num_points, 2))
        labels_InstanceSeg = labels_InstanceSeg.view(-1, 1) # (shape: (batch_size*num_points, 1))
        labels_InstanceSeg = labels_InstanceSeg[:, 0] # (shape: (batch_size*num_points, ))
        loss_InstanceSeg = F.nll_loss(outputs_InstanceSeg, labels_InstanceSeg)
        loss_InstanceSeg_value = loss_InstanceSeg.data.cpu().numpy()
        batch_losses_InstanceSeg.append(loss_InstanceSeg_value)

        ########################################################################
        # compute the TNet loss:
        ########################################################################
        # mask entries corresponding to empty seg point clouds (select only the entries which we care about):
        outputs_TNet = outputs_TNet[dont_care_mask, :] # (shape: (batch_size*, 3))
        labels_TNet = labels_TNet[dont_care_mask, :] # (shape: (batch_size*, 3))
        seg_point_clouds_mean = seg_point_clouds_mean[dont_care_mask, :] # (shape: (batch_size*, 3))

        # shift the GT to the seg point clouds local coords:
        labels_TNet = labels_TNet - seg_point_clouds_mean

        # compute the Huber (smooth L1) loss:
        if outputs_TNet.size()[0] == 0:
            loss_TNet = Variable(torch.from_numpy(np.zeros((1, ), dtype=np.float32))).cuda()
        else:
            loss_TNet = regression_loss_func(outputs_TNet, labels_TNet)

        loss_TNet_value = loss_TNet.data.cpu().numpy()
        batch_losses_TNet.append(loss_TNet_value)

        ########################################################################
        # compute the BboxNet loss:
        ########################################################################
        # mask entries corresponding to empty seg point clouds (select only the entries which we care about):
        outputs_BboxNet = outputs_BboxNet[dont_care_mask, :] # (shape: (batch_size*, 3 + 3 + 2*NH))
        labels_BboxNet = labels_BboxNet[dont_care_mask, :]# (shape: (batch_size*, 11))

        if outputs_BboxNet.size()[0] == 0:
            loss_BboxNet = Variable(torch.from_numpy(np.zeros((1, ), dtype=np.float32))).cuda()
            loss_BboxNet_size = Variable(torch.from_numpy(np.zeros((1, ), dtype=np.float32))).cuda()
            loss_BboxNet_center = Variable(torch.from_numpy(np.zeros((1, ), dtype=np.float32))).cuda()
            loss_BboxNet_heading_class = Variable(torch.from_numpy(np.zeros((1, ), dtype=np.float32))).cuda()
            loss_BboxNet_heading_regr = Variable(torch.from_numpy(np.zeros((1, ), dtype=np.float32))).cuda()
        else:
            # compute the BboxNet center loss:
            labels_BboxNet_center = labels_BboxNet[:, 0:3] # (shape: (batch_size*, 3))
            # # shift the center GT to local coords:
            labels_BboxNet_center = Variable(labels_BboxNet_center.data - seg_point_clouds_mean.data - outputs_TNet.data).cuda() # (outputs_TNet is a variable outputted by model, so it requires grads, which cant be passed as target to the loss function)
            # # compute the Huber (smooth L1) loss:
            outputs_BboxNet_center = outputs_BboxNet[:, 0:3]
            loss_BboxNet_center = regression_loss_func(outputs_BboxNet_center, labels_BboxNet_center)

            # compute the BboxNet size loss:
            labels_BboxNet_size = labels_BboxNet[:, 3:6] # (shape: (batch_size*, 3))
            # # subtract the mean car size in train:
            labels_BboxNet_size = labels_BboxNet_size - labels_BboxNet[:, 8:]
            # # compute the Huber (smooth L1) loss:
            loss_BboxNet_size = regression_loss_func(outputs_BboxNet[:, 3:6], labels_BboxNet_size)

            # compute the BboxNet heading loss
            # # compute the classification loss:
            labels_BboxNet_heading_class = Variable(labels_BboxNet[:, 6].data.type(torch.LongTensor)).cuda() # (shape: (batch_size*, ))
            outputs_BboxNet_heading_class = outputs_BboxNet[:, 6:(6+NH)] # (shape: (batch_size*, NH))
            loss_BboxNet_heading_class = F.nll_loss(F.log_softmax(outputs_BboxNet_heading_class, dim=1), labels_BboxNet_heading_class)
            # # compute the regression loss:
            # # # # get the GT residual for the GT bin:
            labels_BboxNet_heading_regr = labels_BboxNet[:, 7] # (shape: (batch_size*, ))
            # # # # get the pred residual for all bins:
            outputs_BboxNet_heading_regr_all = outputs_BboxNet[:, (6+NH):] # (shape: (batch_size*, 8))
            # # # # get the pred residual for the GT bin:
            outputs_BboxNet_heading_regr = outputs_BboxNet_heading_regr_all.gather(1, labels_BboxNet_heading_class.view(-1, 1)) # (shape: (batch_size*, 1))
            outputs_BboxNet_heading_regr = outputs_BboxNet_heading_regr[:, 0] # (shape: (batch_size*, )
            # # # # compute the loss:
            loss_BboxNet_heading_regr = regression_loss_func(outputs_BboxNet_heading_regr, labels_BboxNet_heading_regr)
            # # compute the total BBoxNet heading loss:
            loss_BboxNet_heading = loss_BboxNet_heading_class + 10*loss_BboxNet_heading_regr

            # compute the BboxNet total loss:
            loss_BboxNet = loss_BboxNet_center + loss_BboxNet_size + loss_BboxNet_heading

        loss_BboxNet_value = loss_BboxNet.data.cpu().numpy()
        batch_losses_BboxNet.append(loss_BboxNet_value)

        loss_BboxNet_size_value = loss_BboxNet_size.data.cpu().numpy()
        batch_losses_BboxNet_size.append(loss_BboxNet_size_value)

        loss_BboxNet_center_value = loss_BboxNet_center.data.cpu().numpy()
        batch_losses_BboxNet_center.append(loss_BboxNet_center_value)

        loss_BboxNet_heading_class_value = loss_BboxNet_heading_class.data.cpu().numpy()
        batch_losses_BboxNet_heading_class.append(loss_BboxNet_heading_class_value)

        loss_BboxNet_heading_regr_value = loss_BboxNet_heading_regr.data.cpu().numpy()
        batch_losses_BboxNet_heading_regr.append(loss_BboxNet_heading_regr_value)

        ########################################################################
        # compute the corner loss:
        ########################################################################
        # mask entries corresponding to empty seg point clouds (select only the entries which we care about):
        labels_corner = labels_corner[dont_care_mask]# (shape: (batch_size*, 8, 3))
        labels_corner_flipped = labels_corner_flipped[dont_care_mask]# (shape: (batch_size*, 8, 3))

        if outputs_BboxNet.size()[0] == 0:
            loss_corner = Variable(torch.from_numpy(np.zeros((1, ), dtype=np.float32))).cuda()
        else:
            outputs_BboxNet_center = outputs_BboxNet[:, 0:3] # (shape: (batch_size*, 3))
            # shift to the same coords used in the labels for the corner loss:
            pred_center = outputs_BboxNet_center + seg_point_clouds_mean + outputs_TNet # (shape: (batch_size*, 3))
            pred_center_unsqeezed = pred_center.unsqueeze(2) # (shape: (batch_size, 3, 1))

            # shift the outputted size to the same "coords" used in the labels for the corner loss:
            outputs_BboxNet_size = outputs_BboxNet[:, 3:6] + labels_BboxNet[:, 8:] # (shape: (batch_size*, 3))

            pred_h = outputs_BboxNet_size[:, 0] # (shape: (batch_size*, ))
            pred_w = outputs_BboxNet_size[:, 1] # (shape: (batch_size*, ))
            pred_l = outputs_BboxNet_size[:, 2] # (shape: (batch_size*, ))

            # get the pred residuals for the GT bins:
            pred_residuals = outputs_BboxNet_heading_regr # (shape: (batch_size*, ))

            Rmat = Variable(torch.zeros(pred_h.size()[0], 3, 3), requires_grad=True).cuda() # (shape: (batch_size*, 3, 3))
            Rmat[:, 0, 0] = torch.cos(pred_residuals)
            Rmat[:, 0, 2] = torch.sin(pred_residuals)
            Rmat[:, 1, 1] = 1
            Rmat[:, 2, 0] = -torch.sin(pred_residuals)
            Rmat[:, 2, 2] = torch.cos(pred_residuals)

            p0_orig = Variable(torch.zeros(pred_h.size()[0], 3, 1), requires_grad=True).cuda() # (shape: (batch_size*, 3, 1))
            p0_orig[:, 0, 0] = pred_l/2.0
            p0_orig[:, 2, 0] = pred_w/2.0

            p1_orig = Variable(torch.zeros(pred_h.size()[0], 3, 1), requires_grad=True).cuda() # (shape: (batch_size*, 3, 1))
            p1_orig[:, 0, 0] = -pred_l/2.0
            p1_orig[:, 2, 0] = pred_w/2.0

            p2_orig = Variable(torch.zeros(pred_h.size()[0], 3, 1), requires_grad=True).cuda() # (shape: (batch_size*, 3, 1))
            p2_orig[:, 0, 0] = -pred_l/2.0
            p2_orig[:, 2, 0] = -pred_w/2.0

            p3_orig = Variable(torch.zeros(pred_h.size()[0], 3, 1), requires_grad=True).cuda() # (shape: (batch_size*, 3, 1))
            p3_orig[:, 0, 0] = pred_l/2.0
            p3_orig[:, 2, 0] = -pred_w/2.0

            p4_orig = Variable(torch.zeros(pred_h.size()[0], 3, 1), requires_grad=True).cuda() # (shape: (batch_size*, 3, 1))
            p4_orig[:, 0, 0] = pred_l/2.0
            p4_orig[:, 1, 0] = -pred_h
            p4_orig[:, 2, 0] = pred_w/2.0

            p5_orig = Variable(torch.zeros(pred_h.size()[0], 3, 1), requires_grad=True).cuda() # (shape: (batch_size*, 3, 1))
            p5_orig[:, 0, 0] = -pred_l/2.0
            p5_orig[:, 1, 0] = -pred_h
            p5_orig[:, 2, 0] = pred_w/2.0

            p6_orig = Variable(torch.zeros(pred_h.size()[0], 3, 1), requires_grad=True).cuda() # (shape: (batch_size*, 3, 1))
            p6_orig[:, 0, 0] = -pred_l/2.0
            p6_orig[:, 1, 0] = -pred_h
            p6_orig[:, 2, 0] = -pred_w/2.0

            p7_orig = Variable(torch.zeros(pred_h.size()[0], 3, 1), requires_grad=True).cuda() # (shape: (batch_size*, 3, 1))
            p7_orig[:, 0, 0] = pred_l/2.0
            p7_orig[:, 1, 0] = -pred_h
            p7_orig[:, 2, 0] = -pred_w/2.0

            pred_p0_unsqeezed = pred_center_unsqeezed + torch.bmm(Rmat, p0_orig) # (shape: (batch_size*, 3, 1))
            pred_p0 = pred_p0_unsqeezed[:, :, 0] # (shape: (batch_size*, 3))
            pred_p1_unsqeezed = pred_center_unsqeezed + torch.bmm(Rmat, p1_orig) # (shape: (batch_size*, 3, 1))
            pred_p1 = pred_p1_unsqeezed[:, :, 0] # (shape: (batch_size*, 3))
            pred_p2_unsqeezed = pred_center_unsqeezed + torch.bmm(Rmat, p2_orig) # (shape: (batch_size*, 3, 1))
            pred_p2 = pred_p2_unsqeezed[:, :, 0] # (shape: (batch_size*, 3))
            pred_p3_unsqeezed = pred_center_unsqeezed + torch.bmm(Rmat, p3_orig) # (shape: (batch_size*, 3, 1))
            pred_p3 = pred_p3_unsqeezed[:, :, 0] # (shape: (batch_size*, 3))
            pred_p4_unsqeezed = pred_center_unsqeezed + torch.bmm(Rmat, p4_orig) # (shape: (batch_size*, 3, 1))
            pred_p4 = pred_p4_unsqeezed[:, :, 0] # (shape: (batch_size*, 3))
            pred_p5_unsqeezed = pred_center_unsqeezed + torch.bmm(Rmat, p5_orig) # (shape: (batch_size*, 3, 1))
            pred_p5 = pred_p5_unsqeezed[:, :, 0] # (shape: (batch_size*, 3))
            pred_p6_unsqeezed = pred_center_unsqeezed + torch.bmm(Rmat, p6_orig) # (shape: (batch_size*, 3, 1))
            pred_p6 = pred_p6_unsqeezed[:, :, 0] # (shape: (batch_size*, 3))
            pred_p7_unsqeezed = pred_center_unsqeezed + torch.bmm(Rmat, p7_orig) # (shape: (batch_size*, 3, 1))
            pred_p7 = pred_p7_unsqeezed[:, :, 0] # (shape: (batch_size*, 3))

            outputs_corner = Variable(torch.zeros(pred_h.size()[0], 8, 3), requires_grad=True).cuda() # (shape: (batch_size*, 8, 3))
            outputs_corner[:, 0] = pred_p0
            outputs_corner[:, 1] = pred_p1
            outputs_corner[:, 2] = pred_p2
            outputs_corner[:, 3] = pred_p3
            outputs_corner[:, 4] = pred_p4
            outputs_corner[:, 5] = pred_p5
            outputs_corner[:, 6] = pred_p6
            outputs_corner[:, 7] = pred_p7

            loss_corner_unflipped = regression_loss_func(outputs_corner, labels_corner)
            loss_corner_flipped = regression_loss_func(outputs_corner, labels_corner_flipped)

            loss_corner = torch.min(loss_corner_unflipped, loss_corner_flipped)

        loss_corner_value = loss_corner.data.cpu().numpy()
        batch_losses_corner.append(loss_corner_value)

        ########################################################################
        # compute the total loss:
        ########################################################################
        lambda_value = 1
        gamma_value = 10
        loss = loss_InstanceSeg + lambda_value*(loss_TNet + loss_BboxNet + gamma_value*loss_corner)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

# compute the val epoch loss:
epoch_loss = np.mean(batch_losses)
print ("validation loss: %g" % epoch_loss)
# compute the val epoch TNet loss:
epoch_loss = np.mean(batch_losses_TNet)
print ("validation TNet loss: %g" % epoch_loss)
# compute the val epoch InstanceSeg loss:
epoch_loss = np.mean(batch_losses_InstanceSeg)
print ("validation InstanceSeg loss: %g" % epoch_loss)
# compute the val epoch BboxNet loss:
epoch_loss = np.mean(batch_losses_BboxNet)
print ("validation BboxNet loss: %g" % epoch_loss)
# compute the val epoch BboxNet size loss:
epoch_loss = np.mean(batch_losses_BboxNet_size)
print ("validation BboxNet size loss: %g" % epoch_loss)
# compute the val epoch BboxNet center loss:
epoch_loss = np.mean(batch_losses_BboxNet_center)
print ("validation BboxNet center loss: %g" % epoch_loss)
# compute the val epoch BboxNet heading class loss:
epoch_loss = np.mean(batch_losses_BboxNet_heading_class)
print ("validation BboxNet heading class loss: %g" % epoch_loss)
# compute the val epoch BboxNet heading regr loss:
epoch_loss = np.mean(batch_losses_BboxNet_heading_regr)
print ("validation BboxNet heading regr loss: %g" % epoch_loss)
# compute the val epoch heading class accuracy:
epoch_accuracy = np.mean(batch_accuracies_heading_class)
print ("validation heading class accuracy: %g" % epoch_accuracy)
# compute the val epoch corner loss:
epoch_loss = np.mean(batch_losses_corner)
print ("validation corner loss: %g" % epoch_loss)
# compute the val epoch accuracy:
epoch_accuracy = np.mean(batch_accuracies)
print ("validation accuracy: %g" % epoch_accuracy)
# compute the val epoch precision:
epoch_precision = np.mean(batch_precisions)
print ("validation precision: %g" % epoch_precision)
# compute the val epoch recall:
epoch_recall = np.mean(batch_recalls)
print ("validation recall: %g" % epoch_recall)
# compute the val epoch f1:
epoch_f1 = np.mean(batch_f1s)
print ("validation f1: %g" % epoch_f1)

with open("%s/eval_dict_val.pkl" % network.model_dir, "wb") as file:
    pickle.dump(eval_dict, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
