# camera-ready

from datasets_imgnet import DatasetImgNetAugmentation, DatasetImgNetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
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

# NOTE! NOTE! change this to not overwrite all log data when you train the model:
model_id = "Image-Only_1"

num_epochs = 1000
batch_size = 8
learning_rate = 0.001

network = ImgNet(model_id, project_dir="/root/3DOD_thesis")
network = network.cuda()

train_dataset = DatasetImgNetAugmentation(kitti_data_path="/root/3DOD_thesis/data/kitti",
                                          kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
                                          type="train")
val_dataset = DatasetImgNetEval(kitti_data_path="/root/3DOD_thesis/data/kitti",
                                kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
                                type="val")

num_train_batches = int(len(train_dataset)/batch_size)
num_val_batches = int(len(val_dataset)/batch_size)

print ("num_train_batches:", num_train_batches)
print ("num_val_batches:", num_val_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=4)

regression_loss_func = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

epoch_losses_train = []
epoch_losses_size_train = []
epoch_losses_keypoints_train = []
epoch_losses_distance_train = []
epoch_losses_val = []
epoch_losses_size_val = []
epoch_losses_keypoints_val = []
epoch_losses_distance_val = []
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
    batch_losses_size = []
    batch_losses_keypoints = []
    batch_losses_distance = []
    for step, (bbox_2d_imgs, labels_size, labels_keypoints, labels_distance) in enumerate(train_loader):
        bbox_2d_imgs = Variable(bbox_2d_imgs) # (shape: (batch_size, 3, H, W) = (batch_size, 3, 224, 224))
        labels_size = Variable(labels_size) # (shape: (batch_size, 3))
        labels_keypoints = Variable(labels_keypoints) # (shape: (batch_size, 2*8))
        labels_distance = Variable(labels_distance) # (shape: (batch_size, 1))
        labels_distance = labels_distance.view(-1) # (shape: (batch_size, ))

        bbox_2d_imgs = bbox_2d_imgs.cuda()
        labels_size = labels_size.cuda()
        labels_keypoints = labels_keypoints.cuda()
        labels_distance = labels_distance.cuda()

        outputs = network(bbox_2d_imgs) # (shape: (batch_size, 2*8 + 3 +1 = 20))
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
        # optimization step:
        ########################################################################
        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print ("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
    plt.close(1)

    epoch_loss = np.mean(batch_losses_size)
    epoch_losses_size_train.append(epoch_loss)
    with open("%s/epoch_losses_size_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_size_train, file)
    print ("train size loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_size_train, "k^")
    plt.plot(epoch_losses_size_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train size loss per epoch")
    plt.savefig("%s/epoch_losses_size_train.png" % network.model_dir)
    plt.close(1)

    epoch_loss = np.mean(batch_losses_keypoints)
    epoch_losses_keypoints_train.append(epoch_loss)
    with open("%s/epoch_losses_keypoints_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_keypoints_train, file)
    print ("train keypoints loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_keypoints_train, "k^")
    plt.plot(epoch_losses_keypoints_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train keypoints loss per epoch")
    plt.savefig("%s/epoch_losses_keypoints_train.png" % network.model_dir)
    plt.close(1)

    epoch_loss = np.mean(batch_losses_distance)
    epoch_losses_distance_train.append(epoch_loss)
    with open("%s/epoch_losses_distance_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_distance_train, file)
    print ("train distance loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_distance_train, "k^")
    plt.plot(epoch_losses_distance_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train distance loss per epoch")
    plt.savefig("%s/epoch_losses_distance_train.png" % network.model_dir)
    plt.close(1)

    print ("####")

    ################################################################################
    # val:
    ################################################################################
    network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    batch_losses_size = []
    batch_losses_keypoints = []
    batch_losses_distance = []
    for step, (bbox_2d_imgs, labels_size, labels_keypoints, labels_distance, img_ids, mean_car_size, ws, hs, u_centers, v_centers, camera_matrices, gt_centers, gt_r_ys, mean_distance) in enumerate(val_loader):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            bbox_2d_imgs = Variable(bbox_2d_imgs) # (shape: (batch_size, 3, H, W) = (batch_size, 3, 224, 224))
            labels_size = Variable(labels_size) # (shape: (batch_size, 3))
            labels_keypoints = Variable(labels_keypoints) # (shape: (batch_size, 2*8))
            labels_distance = Variable(labels_distance) # (shape: (batch_size, 1))
            labels_distance = labels_distance.view(-1) # (shape: (batch_size, ))

            bbox_2d_imgs = bbox_2d_imgs.cuda()
            labels_size = labels_size.cuda()
            labels_keypoints = labels_keypoints.cuda()
            labels_distance = labels_distance.cuda()

            outputs = network(bbox_2d_imgs) # (shape: (batch_size, 2*8 + 3 + 1 = 20))
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

            # if step == 1:
            #     output_keypoints = outputs_keypoints[0]
            #     label_keypoints = labels_keypoints[0]
            #     w = ws[0]
            #     h = hs[0]
            #     u_center = u_centers[0]
            #     v_center = v_centers[0]
            #     img_id = img_ids[0]
            #
            #     img = cv2.imread(val_dataset.img_dir + img_id + ".png", -1)
            #
            #     label_keypoints = np.resize(label_keypoints, (8, 2))
            #     label_keypoints = label_keypoints*np.array([w, h]) + np.array([u_center, v_center])
            #     img_with_gt_3dbbox = draw_3dbbox_from_keypoints(img, label_keypoints)
            #     cv2.imwrite("%s/img_with_gt_3dbbox_epoch_%d.png" % (network.model_dir, epoch+1), img_with_gt_3dbbox)
            #
            #     output_keypoints = np.resize(output_keypoints, (8, 2))
            #     output_keypoints = output_keypoints*np.array([w, h]) + np.array([u_center, v_center])
            #     img_with_pred_3dbbox = draw_3dbbox_from_keypoints(img, output_keypoints)
            #     cv2.imwrite("%s/img_with_pred_3dbbox_epoch_%d.png" % (network.model_dir, epoch+1), img_with_pred_3dbbox)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print ("val loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_val, "k^")
    plt.plot(epoch_losses_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val loss per epoch")
    plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
    plt.close(1)

    epoch_loss = np.mean(batch_losses_size)
    epoch_losses_size_val.append(epoch_loss)
    with open("%s/epoch_losses_size_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_size_val, file)
    print ("val size loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_size_val, "k^")
    plt.plot(epoch_losses_size_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val size loss per epoch")
    plt.savefig("%s/epoch_losses_size_val.png" % network.model_dir)
    plt.close(1)

    epoch_loss = np.mean(batch_losses_keypoints)
    epoch_losses_keypoints_val.append(epoch_loss)
    with open("%s/epoch_losses_keypoints_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_keypoints_val, file)
    print ("val keypoints loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_keypoints_val, "k^")
    plt.plot(epoch_losses_keypoints_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val keypoints loss per epoch")
    plt.savefig("%s/epoch_losses_keypoints_val.png" % network.model_dir)
    plt.close(1)

    epoch_loss = np.mean(batch_losses_distance)
    epoch_losses_distance_val.append(epoch_loss)
    with open("%s/epoch_losses_distance_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_distance_val, file)
    print ("val distance loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_distance_val, "k^")
    plt.plot(epoch_losses_distance_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val distance loss per epoch")
    plt.savefig("%s/epoch_losses_distance_val.png" % network.model_dir)
    plt.close(1)

    # save the model weights to disk:
    checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)
