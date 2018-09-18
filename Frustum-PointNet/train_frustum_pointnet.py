# camera-ready

from datasets import DatasetFrustumPointNetAugmentation, EvalDatasetFrustumPointNet, getBinCenters, wrapToPi # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
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

# NOTE! NOTE! change this to not overwrite all log data when you train the model:
model_id = "Frustum-PointNet_1"

num_epochs = 700
batch_size = 32
learning_rate = 0.001

network = FrustumPointNet(model_id, project_dir="/root/3DOD_thesis")
network = network.cuda()

NH = network.BboxNet_network.NH

train_dataset = DatasetFrustumPointNetAugmentation(kitti_data_path="/root/3DOD_thesis/data/kitti",
                                                   kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
                                                   type="train", NH=NH)
val_dataset = EvalDatasetFrustumPointNet(kitti_data_path="/root/3DOD_thesis/data/kitti",
                                         kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
                                         type="val", NH=NH)

num_train_batches = int(len(train_dataset)/batch_size)
num_val_batches = int(len(val_dataset)/batch_size)

print ("num_train_batches:", num_train_batches)
print ("num_val_batches:", num_val_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=16)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=16)

regression_loss_func = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

epoch_losses_train = []
epoch_losses_InstanceSeg_train = []
epoch_losses_TNet_train = []
epoch_losses_BboxNet_train = []
epoch_losses_BboxNet_size_train = []
epoch_losses_BboxNet_center_train = []
epoch_losses_BboxNet_heading_class_train = []
epoch_losses_BboxNet_heading_regr_train = []
epoch_accuracies_heading_class_train = []
epoch_losses_corner_train = []
epoch_accuracies_train = []
epoch_precisions_train = []
epoch_recalls_train = []
epoch_f1s_train = []
epoch_losses_val = []
epoch_losses_InstanceSeg_val = []
epoch_losses_TNet_val = []
epoch_losses_BboxNet_val = []
epoch_losses_BboxNet_size_val = []
epoch_losses_BboxNet_center_val = []
epoch_losses_BboxNet_heading_class_val = []
epoch_losses_BboxNet_heading_regr_val = []
epoch_accuracies_heading_class_val = []
epoch_losses_corner_val = []
epoch_accuracies_val = []
epoch_precisions_val = []
epoch_recalls_val = []
epoch_f1s_val = []
for epoch in range(num_epochs):
    print ("###########################")
    print ("######## NEW EPOCH ########")
    print ("###########################")
    print ("epoch: %d/%d" % (epoch+1, num_epochs))

    if epoch % 100 == 0 and epoch > 0:
        learning_rate = learning_rate/2
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    print ("learning_rate:")
    print (learning_rate)

    ################################################################################
    # train:
    ################################################################################
    network.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    batch_losses_InstanceSeg = []
    batch_losses_TNet = []
    batch_losses_BboxNet = []
    batch_losses_BboxNet_center = []
    batch_losses_BboxNet_size = []
    batch_losses_BboxNet_heading_regr = []
    batch_losses_BboxNet_heading_class = []
    batch_losses_corner = []
    batch_accuracies = []
    batch_precisions = []
    batch_recalls = []
    batch_f1s = []
    batch_accuracies_heading_class = []
    for step, (frustum_point_clouds, labels_InstanceSeg, labels_TNet, labels_BboxNet, labels_corner, labels_corner_flipped) in enumerate(train_loader):
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
            ####
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

        ########################################################################
        # optimization step:
        ########################################################################
        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)

    # compute the train epoch loss:
    epoch_loss = np.mean(batch_losses)
    # save the train epoch loss:
    epoch_losses_train.append(epoch_loss)
    # save the train epoch loss to disk:
    with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print ("training loss: %g" % epoch_loss)
    # plot the training loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("training loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch TNet loss:
    epoch_loss = np.mean(batch_losses_TNet)
    # save the train epoch loss:
    epoch_losses_TNet_train.append(epoch_loss)
    # save the train epoch TNet loss to disk:
    with open("%s/epoch_losses_TNet_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_TNet_train, file)
    print ("training TNet loss: %g" % epoch_loss)
    # plot the training TNet loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_TNet_train, "k^")
    plt.plot(epoch_losses_TNet_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("training TNet loss per epoch")
    plt.savefig("%s/epoch_losses_TNet_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch InstanceSeg loss:
    epoch_loss = np.mean(batch_losses_InstanceSeg)
    # save the train epoch loss:
    epoch_losses_InstanceSeg_train.append(epoch_loss)
    # save the train epoch InstanceSeg loss to disk:
    with open("%s/epoch_losses_InstanceSeg_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_InstanceSeg_train, file)
    print ("training InstanceSeg loss: %g" % epoch_loss)
    # plot the training InstanceSeg loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_InstanceSeg_train, "k^")
    plt.plot(epoch_losses_InstanceSeg_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("training InstanceSeg loss per epoch")
    plt.savefig("%s/epoch_losses_InstanceSeg_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch BboxNet loss:
    epoch_loss = np.mean(batch_losses_BboxNet)
    # save the train epoch loss:
    epoch_losses_BboxNet_train.append(epoch_loss)
    # save the train epoch BboxNet loss to disk:
    with open("%s/epoch_losses_BboxNet_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_BboxNet_train, file)
    print ("training BboxNet loss: %g" % epoch_loss)
    # plot the training BboxNet loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_BboxNet_train, "k^")
    plt.plot(epoch_losses_BboxNet_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("training BboxNet loss per epoch")
    plt.savefig("%s/epoch_losses_BboxNet_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch BboxNet size loss:
    epoch_loss = np.mean(batch_losses_BboxNet_size)
    # save the train epoch loss:
    epoch_losses_BboxNet_size_train.append(epoch_loss)
    # save the train epoch BboxNet size loss to disk:
    with open("%s/epoch_losses_BboxNet_size_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_BboxNet_size_train, file)
    print ("training BboxNet size loss: %g" % epoch_loss)
    # plot the training BboxNet size loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_BboxNet_size_train, "k^")
    plt.plot(epoch_losses_BboxNet_size_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("training BboxNet size loss per epoch")
    plt.savefig("%s/epoch_losses_BboxNet_size_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch BboxNet center loss:
    epoch_loss = np.mean(batch_losses_BboxNet_center)
    # save the train epoch loss:
    epoch_losses_BboxNet_center_train.append(epoch_loss)
    # save the train epoch BboxNet center loss to disk:
    with open("%s/epoch_losses_BboxNet_center_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_BboxNet_center_train, file)
    print ("training BboxNet center loss: %g" % epoch_loss)
    # plot the training BboxNet center loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_BboxNet_center_train, "k^")
    plt.plot(epoch_losses_BboxNet_center_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("training BboxNet center loss per epoch")
    plt.savefig("%s/epoch_losses_BboxNet_center_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch BboxNet heading class loss:
    epoch_loss = np.mean(batch_losses_BboxNet_heading_class)
    # save the train epoch loss:
    epoch_losses_BboxNet_heading_class_train.append(epoch_loss)
    # save the train epoch BboxNet heading class loss to disk:
    with open("%s/epoch_losses_BboxNet_heading_class_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_BboxNet_heading_class_train, file)
    print ("training BboxNet heading class loss: %g" % epoch_loss)
    # plot the training BboxNet heading class loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_BboxNet_heading_class_train, "k^")
    plt.plot(epoch_losses_BboxNet_heading_class_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("training BboxNet heading class loss per epoch")
    plt.savefig("%s/epoch_losses_BboxNet_heading_class_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch BboxNet heading regr loss:
    epoch_loss = np.mean(batch_losses_BboxNet_heading_regr)
    # save the train epoch loss:
    epoch_losses_BboxNet_heading_regr_train.append(epoch_loss)
    # save the train epoch BboxNet heading regr loss to disk:
    with open("%s/epoch_losses_BboxNet_heading_regr_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_BboxNet_heading_regr_train, file)
    print ("training BboxNet heading regr loss: %g" % epoch_loss)
    # plot the training BboxNet heading regr loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_BboxNet_heading_regr_train, "k^")
    plt.plot(epoch_losses_BboxNet_heading_regr_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("training BboxNet heading regr loss per epoch")
    plt.savefig("%s/epoch_losses_BboxNet_heading_regr_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch heading class accuracy:
    epoch_accuracy = np.mean(batch_accuracies_heading_class)
    # save the train epoch heading class accuracy:
    epoch_accuracies_heading_class_train.append(epoch_accuracy)
    # save the train epoch heading class accuracy to disk:
    with open("%s/epoch_accuracies_heading_class_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_accuracies_heading_class_train, file)
    print ("training heading class accuracy: %g" % epoch_accuracy)
    # plot the training heading class accuracy vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_accuracies_heading_class_train, "k^")
    plt.plot(epoch_accuracies_heading_class_train, "k")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title("training heading class accuracy per epoch")
    plt.savefig("%s/epoch_accuracies_heading_class_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch corner loss:
    epoch_loss = np.mean(batch_losses_corner)
    # save the train epoch corner loss:
    epoch_losses_corner_train.append(epoch_loss)
    # save the train epoch corner loss to disk:
    with open("%s/epoch_losses_corner_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_corner_train, file)
    print ("training corner loss: %g" % epoch_loss)
    # plot the training corner loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_corner_train, "k^")
    plt.plot(epoch_losses_corner_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("training corner loss per epoch")
    plt.savefig("%s/epoch_losses_corner_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch accuracy:
    epoch_accuracy = np.mean(batch_accuracies)
    # save the train epoch accuracy:
    epoch_accuracies_train.append(epoch_accuracy)
    # save the train epoch accuracy to disk:
    with open("%s/epoch_accuracies_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_accuracies_train, file)
    print ("training accuracy: %g" % epoch_accuracy)
    # plot the training accuracy vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_accuracies_train, "k^")
    plt.plot(epoch_accuracies_train, "k")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title("training accuracy per epoch")
    plt.savefig("%s/epoch_accuracies_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch precision:
    epoch_precision = np.mean(batch_precisions)
    # save the train epoch precision:
    epoch_precisions_train.append(epoch_precision)
    # save the train epoch precision to disk:
    with open("%s/epoch_precisions_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_precisions_train, file)
    print ("training precision: %g" % epoch_precision)
    # plot the training precision vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_precisions_train, "k^")
    plt.plot(epoch_precisions_train, "k")
    plt.ylabel("precision")
    plt.xlabel("epoch")
    plt.title("training precision per epoch")
    plt.savefig("%s/epoch_precisions_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch recall:
    epoch_recall = np.mean(batch_recalls)
    # save the train epoch recall:
    epoch_recalls_train.append(epoch_recall)
    # save the train epoch recall to disk:
    with open("%s/epoch_recalls_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_recalls_train, file)
    print ("training recall: %g" % epoch_recall)
    # plot the training recall vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_recalls_train, "k^")
    plt.plot(epoch_recalls_train, "k")
    plt.ylabel("recall")
    plt.xlabel("epoch")
    plt.title("training recall per epoch")
    plt.savefig("%s/epoch_recalls_train.png" % network.model_dir)
    plt.close(1)

    # compute the train epoch f1:
    epoch_f1 = np.mean(batch_f1s)
    # save the train epoch f1:
    epoch_f1s_train.append(epoch_f1)
    # save the train epoch f1 to disk:
    with open("%s/epoch_f1s_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_f1s_train, file)
    print ("training f1: %g" % epoch_f1)
    # plot the training f1 vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_f1s_train, "k^")
    plt.plot(epoch_f1s_train, "k")
    plt.ylabel("f1")
    plt.xlabel("epoch")
    plt.title("training f1 per epoch")
    plt.savefig("%s/epoch_f1s_train.png" % network.model_dir)
    plt.close(1)

    print ("####")

    ############################################################################
    # val:
    ############################################################################
    network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    batch_losses_InstanceSeg = []
    batch_losses_TNet = []
    batch_losses_BboxNet = []
    batch_losses_BboxNet_center = []
    batch_losses_BboxNet_size = []
    batch_losses_BboxNet_heading_regr = []
    batch_losses_BboxNet_heading_class = []
    batch_losses_corner = []
    batch_accuracies = []
    batch_precisions = []
    batch_recalls = []
    batch_f1s = []
    batch_accuracies_heading_class = []
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
                ####
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
    # save the val epoch loss:
    epoch_losses_val.append(epoch_loss)
    # save the val epoch loss to disk:
    with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print ("validation loss: %g" % epoch_loss)
    # plot the val loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_val, "k^")
    plt.plot(epoch_losses_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("validation loss per epoch")
    plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch TNet loss:
    epoch_loss = np.mean(batch_losses_TNet)
    # save the val epoch loss:
    epoch_losses_TNet_val.append(epoch_loss)
    # save the val epoch TNet loss to disk:
    with open("%s/epoch_losses_TNet_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_TNet_val, file)
    print ("validation TNet loss: %g" % epoch_loss)
    # plot the validation TNet loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_TNet_val, "k^")
    plt.plot(epoch_losses_TNet_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("validation TNet loss per epoch")
    plt.savefig("%s/epoch_losses_TNet_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch InstanceSeg loss:
    epoch_loss = np.mean(batch_losses_InstanceSeg)
    # save the val epoch loss:
    epoch_losses_InstanceSeg_val.append(epoch_loss)
    # save the val epoch InstanceSeg loss to disk:
    with open("%s/epoch_losses_InstanceSeg_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_InstanceSeg_val, file)
    print ("validation InstanceSeg loss: %g" % epoch_loss)
    # plot the validation InstanceSeg loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_InstanceSeg_val, "k^")
    plt.plot(epoch_losses_InstanceSeg_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("validation InstanceSeg loss per epoch")
    plt.savefig("%s/epoch_losses_InstanceSeg_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch BboxNet loss:
    epoch_loss = np.mean(batch_losses_BboxNet)
    # save the val epoch loss:
    epoch_losses_BboxNet_val.append(epoch_loss)
    # save the val epoch BboxNet loss to disk:
    with open("%s/epoch_losses_BboxNet_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_BboxNet_val, file)
    print ("validation BboxNet loss: %g" % epoch_loss)
    # plot the validation BboxNet loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_BboxNet_val, "k^")
    plt.plot(epoch_losses_BboxNet_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("validation BboxNet loss per epoch")
    plt.savefig("%s/epoch_losses_BboxNet_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch BboxNet size loss:
    epoch_loss = np.mean(batch_losses_BboxNet_size)
    # save the val epoch loss:
    epoch_losses_BboxNet_size_val.append(epoch_loss)
    # save the val epoch BboxNet size loss to disk:
    with open("%s/epoch_losses_BboxNet_size_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_BboxNet_size_val, file)
    print ("validation BboxNet size loss: %g" % epoch_loss)
    # plot the validation BboxNet size loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_BboxNet_size_val, "k^")
    plt.plot(epoch_losses_BboxNet_size_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("validation BboxNet size loss per epoch")
    plt.savefig("%s/epoch_losses_BboxNet_size_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch BboxNet center loss:
    epoch_loss = np.mean(batch_losses_BboxNet_center)
    # save the val epoch loss:
    epoch_losses_BboxNet_center_val.append(epoch_loss)
    # save the val epoch BboxNet center loss to disk:
    with open("%s/epoch_losses_BboxNet_center_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_BboxNet_center_val, file)
    print ("validation BboxNet center loss: %g" % epoch_loss)
    # plot the validation BboxNet center loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_BboxNet_center_val, "k^")
    plt.plot(epoch_losses_BboxNet_center_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("validation BboxNet center loss per epoch")
    plt.savefig("%s/epoch_losses_BboxNet_center_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch BboxNet heading class loss:
    epoch_loss = np.mean(batch_losses_BboxNet_heading_class)
    # save the val epoch loss:
    epoch_losses_BboxNet_heading_class_val.append(epoch_loss)
    # save the val epoch BboxNet heading class loss to disk:
    with open("%s/epoch_losses_BboxNet_heading_class_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_BboxNet_heading_class_val, file)
    print ("validation BboxNet heading class loss: %g" % epoch_loss)
    # plot the validation BboxNet heading class loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_BboxNet_heading_class_val, "k^")
    plt.plot(epoch_losses_BboxNet_heading_class_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("validation BboxNet heading class loss per epoch")
    plt.savefig("%s/epoch_losses_BboxNet_heading_class_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch BboxNet heading regr loss:
    epoch_loss = np.mean(batch_losses_BboxNet_heading_regr)
    # save the val epoch loss:
    epoch_losses_BboxNet_heading_regr_val.append(epoch_loss)
    # save the val epoch BboxNet heading regr loss to disk:
    with open("%s/epoch_losses_BboxNet_heading_regr_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_BboxNet_heading_regr_val, file)
    print ("validation BboxNet heading regr loss: %g" % epoch_loss)
    # plot the validation BboxNet heading regr loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_BboxNet_heading_regr_val, "k^")
    plt.plot(epoch_losses_BboxNet_heading_regr_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("validation BboxNet heading regr loss per epoch")
    plt.savefig("%s/epoch_losses_BboxNet_heading_regr_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch heading class accuracy:
    epoch_accuracy = np.mean(batch_accuracies_heading_class)
    # save the val epoch heading class accuracy:
    epoch_accuracies_heading_class_val.append(epoch_accuracy)
    # save the val epoch heading class accuracy to disk:
    with open("%s/epoch_accuracies_heading_class_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_accuracies_heading_class_val, file)
    print ("validation heading class accuracy: %g" % epoch_accuracy)
    # plot the validation heading class accuracy vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_accuracies_heading_class_val, "k^")
    plt.plot(epoch_accuracies_heading_class_val, "k")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title("validation heading class accuracy per epoch")
    plt.savefig("%s/epoch_accuracies_heading_class_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch corner loss:
    epoch_loss = np.mean(batch_losses_corner)
    # save the val epoch corner loss:
    epoch_losses_corner_val.append(epoch_loss)
    # save the val epoch corner loss to disk:
    with open("%s/epoch_losses_corner_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_corner_val, file)
    print ("validation corner loss: %g" % epoch_loss)
    # plot the validation corner loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_losses_corner_val, "k^")
    plt.plot(epoch_losses_corner_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("validation corner loss per epoch")
    plt.savefig("%s/epoch_losses_corner_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch accuracy:
    epoch_accuracy = np.mean(batch_accuracies)
    # save the val epoch accuracy:
    epoch_accuracies_val.append(epoch_accuracy)
    # save the val epoch accuracy to disk:
    with open("%s/epoch_accuracies_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_accuracies_val, file)
    print ("validation accuracy: %g" % epoch_accuracy)
    # plot the validation accuracy vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_accuracies_val, "k^")
    plt.plot(epoch_accuracies_val, "k")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title("validation accuracy per epoch")
    plt.savefig("%s/epoch_accuracies_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch precision:
    epoch_precision = np.mean(batch_precisions)
    # save the val epoch precision:
    epoch_precisions_val.append(epoch_precision)
    # save the val epoch precision to disk:
    with open("%s/epoch_precisions_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_precisions_val, file)
    print ("validation precision: %g" % epoch_precision)
    # plot the validation precision vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_precisions_val, "k^")
    plt.plot(epoch_precisions_val, "k")
    plt.ylabel("precision")
    plt.xlabel("epoch")
    plt.title("validation precision per epoch")
    plt.savefig("%s/epoch_precisions_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch recall:
    epoch_recall = np.mean(batch_recalls)
    # save the val epoch recall:
    epoch_recalls_val.append(epoch_recall)
    # save the val epoch recall to disk:
    with open("%s/epoch_recalls_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_recalls_val, file)
    print ("validation recall: %g" % epoch_recall)
    # plot the validation recall vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_recalls_val, "k^")
    plt.plot(epoch_recalls_val, "k")
    plt.ylabel("recall")
    plt.xlabel("epoch")
    plt.title("validation recall per epoch")
    plt.savefig("%s/epoch_recalls_val.png" % network.model_dir)
    plt.close(1)

    # compute the val epoch f1:
    epoch_f1 = np.mean(batch_f1s)
    # save the val epoch f1:
    epoch_f1s_val.append(epoch_f1)
    # save the val epoch f1 to disk:
    with open("%s/epoch_f1s_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_f1s_val, file)
    print ("validation f1: %g" % epoch_f1)
    # plot the validation f1 vs epoch and save to disk:
    plt.figure(1)
    plt.plot(epoch_f1s_val, "k^")
    plt.plot(epoch_f1s_val, "k")
    plt.ylabel("f1")
    plt.xlabel("epoch")
    plt.title("validation f1 per epoch")
    plt.savefig("%s/epoch_f1s_val.png" % network.model_dir)
    plt.close(1)

    # save the model weights to disk:
    checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)
