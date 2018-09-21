# camera-ready

# this file contains code snippets which I have found (more or less) useful at
# some point during the project. Probably nothing interesting to see here.







# import pickle
# import numpy as np
#
# with open("/home/fregu856/exjobb/training_logs/imgnet/model_7/epoch_losses_val.pkl", "rb") as file:
#     val_loss = pickle.load(file)
#
# with open("/home/fregu856/exjobb/training_logs/imgnet/model_7/epoch_losses_distance_val.pkl", "rb") as file:
#     val_distance_loss = pickle.load(file)
#
# with open("/home/fregu856/exjobb/training_logs/imgnet/model_7/epoch_losses_keypoints_val.pkl", "rb") as file:
#     val_keypoints_loss = pickle.load(file)
#
# with open("/home/fregu856/exjobb/training_logs/imgnet/model_7/epoch_losses_size_val.pkl", "rb") as file:
#     val_size_loss = pickle.load(file)
#
# print ("val loss min:", np.argmin(np.array(val_loss)), np.min(np.array(val_loss)))
#
# print ("val keypoints loss min:", np.argmin(np.array(val_keypoints_loss)), np.min(np.array(val_keypoints_loss)))
#
# print ("val distance loss min:", np.argmin(np.array(val_distance_loss)), np.min(np.array(val_distance_loss)))
#
# print ("val size loss min:", np.argmin(np.array(val_size_loss)), np.min(np.array(val_size_loss)))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import numpy as np
# import pickle
# import random
#
# with open("/home/fregu856/exjobb/data/kitti/meta/train_img_ids.pkl", "rb") as file:
#     old_train_img_ids = pickle.load(file)
#
# with open("/home/fregu856/exjobb/data/kitti/meta/val_img_ids.pkl", "rb") as file:
#     old_val_img_ids = pickle.load(file)
#
# print ("old_train_img_ids:", len(old_train_img_ids))
# print ("old_val_img_ids:", len(old_val_img_ids))
#
# img_ids = old_train_img_ids + old_val_img_ids
# random.shuffle(img_ids)
# random.shuffle(img_ids)
# random.shuffle(img_ids)
# random.shuffle(img_ids)
#
# no_of_imgs = len(img_ids)
# train_img_ids = img_ids[:int(no_of_imgs*0.9)]
# val_img_ids = img_ids[-int(no_of_imgs*0.1):]
#
# print (no_of_imgs)
# print ("train:", len(train_img_ids))
# print ("val:", len(val_img_ids))
#
# with open("/home/fregu856/exjobb/data/kitti/meta/train_img_ids_random.pkl", "wb") as file:
#     pickle.dump(train_img_ids, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
#
# with open("/home/fregu856/exjobb/data/kitti/meta/val_img_ids_random.pkl", "wb") as file:
#     pickle.dump(val_img_ids, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
# import numpy as np
#
# model_id = "40"
#
# with open("/home/fregu856/exjobb/training_logs/frustum_pointnet/model_" + model_id + "/epoch_losses_TNet_val.pkl", "rb") as file:
#     TNet_val = pickle.load(file)
#
# with open("/home/fregu856/exjobb/training_logs/frustum_pointnet/model_" + model_id + "/epoch_losses_BboxNet_size_val.pkl", "rb") as file:
#     size_val = pickle.load(file)
#
# with open("/home/fregu856/exjobb/training_logs/frustum_pointnet/model_" + model_id + "/epoch_losses_BboxNet_center_val.pkl", "rb") as file:
#     center_val = pickle.load(file)
#
# with open("/home/fregu856/exjobb/training_logs/frustum_pointnet/model_" + model_id + "/epoch_accuracies_heading_class_val.pkl", "rb") as file:
#     data_acc = pickle.load(file)
#
# with open("/home/fregu856/exjobb/training_logs/frustum_pointnet/model_" + model_id + "/epoch_losses_val.pkl", "rb") as file:
#     data_loss = pickle.load(file)
#
# # for index, value in enumerate(data_acc):
# #     print(index, value)
# #
# # for index, value in enumerate(data_loss):
# #     print(index, value)
# #
# # for index, value in enumerate(size_val):
# #     print(index, value)
#
# plt.figure(1)
# plt.plot(data_acc[50:], "k^")
# plt.plot(data_acc[50:], "k")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.title("training heading class accuracy per epoch")
# plt.savefig("test.png")
# plt.close(1)
#
# plt.figure(1)
# plt.plot(data_loss[50:], "k^")
# plt.plot(data_loss[50:], "k")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.title("training heading class accuracy per epoch")
# plt.savefig("test2.png")
# plt.close(1)
#
# print ("heading class accuracy max:", np.argmax(np.array(data_acc)), np.max(np.array(data_acc)))
#
# print ("val loss min:", np.argmin(np.array(data_loss)),  np.min(np.array(data_loss)))
#
# print ("TNet val loss min:", np.argmin(np.array(TNet_val)), np.min(np.array(TNet_val)))
#
# print ("center val loss min:", np.argmin(np.array(center_val)), np.min(np.array(center_val)))
#
# print ("size val loss min:", np.argmin(np.array(size_val)), np.min(np.array(size_val)))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
# import numpy as np
#
# model_id = "10_2"
#
# with open("/home/fregu856/exjobb/training_logs/imgnet/model_" + model_id + "/epoch_losses_val.pkl", "rb") as file:
#     val_loss = pickle.load(file)
#
# print ("val loss min:", np.argmin(np.array(val_loss)),  np.min(np.array(val_loss)))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import numpy as np
# import pickle
#
# with open("/home/fregu856/exjobb/data/kitti/meta/val_img_ids_random.pkl", "rb") as file: # (needed for python3)
#     val_img_ids = pickle.load(file)
#
# print len(val_img_ids)
#
# with open("/home/fregu856/exjobb/code/eval_kitti/build/lists/val_random.txt", "w") as txt_file:
#     for img_id in val_img_ids:
#         txt_file.write(img_id + "\n")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# ################################################################################
# # split SYN into train/val and save to disk:
# ################################################################################
# # NOTE NOTE needs to be run with python3
#
# import numpy as np
# import random
# import pickle
#
# # SYN has 25000 images, ids: {1, 2, 3,..., 25000}
#
# img_ids = list(np.arange(1,25000+1))
#
# random.shuffle(img_ids)
# random.shuffle(img_ids)
# random.shuffle(img_ids)
#
# no_of_imgs = len(img_ids)
# train_img_ids = img_ids[:int(no_of_imgs*0.8)]
# val_img_ids = img_ids[-int(no_of_imgs*0.2):]
#
# with open("/home/fregu856/exjobb/data/7dlabs/meta/train_img_ids.pkl", "wb") as file:
#     pickle.dump(train_img_ids, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
#
# with open("/home/fregu856/exjobb/data/7dlabs/meta/val_img_ids.pkl", "wb") as file:
#     pickle.dump(val_img_ids, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
#
#
#
#
#
#
#
#
#
#
#
#
# ################################################################################
# # compute mean of all centered/rotated input frustum point clouds in train:
# ################################################################################
# from datasets import DatasetFrustumPointNet # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
# from frustum_pointnet import FrustumPointNet
#
# import torch
# import torch.utils.data
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
# import torch.nn.functional as F
#
# import numpy as np
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
# batch_size = 32
#
# train_dataset = DatasetFrustumPointNet(kitti_data_path="/datasets/kitti",
#                                        kitti_meta_path="/staging/frexgus/kitti/meta",
#                                        type="train", NH=4)
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=16)
#
# mean_xyz_list = []
# for step, (centered_frustum_point_clouds_camera, labels_InstanceSeg, labels_TNet, labels_BboxNet, labels_corner, labels_corner_flipped) in enumerate(train_loader):
#
#     frustum_point_clouds = centered_frustum_point_clouds_camera.numpy()
#     frustum_point_clouds_xyz = frustum_point_clouds[:, :, 0:3]
#
#     mean_xyz = np.mean(frustum_point_clouds_xyz, 1) # (shape: (batch_size, 3))
#     mean_xyz = np.mean(mean_xyz, 0) # (shape: (3, ))
#
#     mean_xyz_list.append(mean_xyz)
#
# num_means = len(mean_xyz_list)
#
# means = np.zeros((num_means, 3))
# for i in range(num_means):
#     means[i] = mean_xyz_list[i]
#
# mean_xyz = np.mean(means, 0)
#
# print (mean_xyz)
#
# with open("/staging/frexgus/kitti/meta/centered_frustum_mean_xyz.pkl", "wb") as file:
#   pickle.dump(mean_xyz, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import pickle
# import numpy
#
# with open("/home/fregu856/exjobb/data/7dlabs/meta/kitti_train_mean_car_size.pkl", "rb") as file:
#     data = pickle.load(file)
#
# print (data)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import os
# import numpy as np
# import cPickle
#
# train_file_path = "/home/fregu856/exjobb/data/kitti/meta/train.txt"
# val_file_path = "/home/fregu856/exjobb/data/kitti/meta/val.txt"
#
# train_img_ids = []
# with open(train_file_path) as train_file:
#     for line in train_file:
#         img_id = line.strip()
#         train_img_ids.append(img_id)
#
# val_img_ids = []
# with open(val_file_path) as val_file:
#     for line in val_file:
#         img_id = line.strip()
#         val_img_ids.append(img_id)
#
# cPickle.dump(train_img_ids, open("/home/fregu856/exjobb/data/kitti/meta/train_img_ids.pkl", "w"))
# cPickle.dump(val_img_ids, open("/home/fregu856/exjobb/data/kitti/meta/val_img_ids.pkl", "w"))
#
# # train_img_ids = cPickle.load(open("/home/fregu856/exjobb/data/kitti/meta/train_img_ids.pkl"))
# # val_img_ids = cPickle.load(open("/home/fregu856/exjobb/data/kitti/meta/val_img_ids.pkl"))
# #
# # print len(train_img_ids)
# # print len(val_img_ids)
