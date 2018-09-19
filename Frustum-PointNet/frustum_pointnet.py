# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import numpy as np

class InstanceSeg(nn.Module):
    def __init__(self, num_points=1024):
        super(InstanceSeg, self).__init__()

        self.num_points = num_points

        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.conv6 = nn.Conv1d(1088, 512, 1)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, 128, 1)
        self.conv10 = nn.Conv1d(128, 2, 1)
        self.max_pool = nn.MaxPool1d(num_points)

    def forward(self, x):
        batch_size = x.size()[0] # (x has shape (batch_size, 4, num_points))

        out = F.relu(self.conv1(x)) # (shape: (batch_size, 64, num_points))
        out = F.relu(self.conv2(out)) # (shape: (batch_size, 64, num_points))
        point_features = out

        out = F.relu(self.conv3(out)) # (shape: (batch_size, 64, num_points))
        out = F.relu(self.conv4(out)) # (shape: (batch_size, 128, num_points))
        out = F.relu(self.conv5(out)) # (shape: (batch_size, 1024, num_points))
        global_feature = self.max_pool(out) # (shape: (batch_size, 1024, 1))

        global_feature_repeated = global_feature.repeat(1, 1, self.num_points) # (shape: (batch_size, 1024, num_points))
        out = torch.cat([global_feature_repeated, point_features], 1) # (shape: (batch_size, 1024+64=1088, num_points))

        out = F.relu(self.conv6(out)) # (shape: (batch_size, 512, num_points))
        out = F.relu(self.conv7(out)) # (shape: (batch_size, 256, num_points))
        out = F.relu(self.conv8(out)) # (shape: (batch_size, 128, num_points))
        out = F.relu(self.conv9(out)) # (shape: (batch_size, 128, num_points))

        out = self.conv10(out) # (shape: (batch_size, 2, num_points))

        out = out.transpose(2,1).contiguous() # (shape: (batch_size, num_points, 2))
        out = F.log_softmax(out.view(-1, 2), dim=1) # (shape: (batch_size*num_points, 2))
        out = out.view(batch_size, self.num_points, 2) # (shape: (batch_size, num_points, 2))

        return out

class TNet(nn.Module):
    def __init__(self, num_seg_points=512):
        super(TNet, self).__init__()

        self.num_seg_points = num_seg_points

        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.max_pool = nn.MaxPool1d(num_seg_points)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        # (x has shape (batch_size, 3, num_seg_points))

        out = F.relu(self.conv1(x)) # (shape: (batch_size, 128, num_seg_points))
        out = F.relu(self.conv2(out)) # (shape: (batch_size, 256, num_seg_points))
        out = F.relu(self.conv3(out)) # (shape: (batch_size, 512, num_seg_points))
        out = self.max_pool(out) # (shape: (batch_size, 512, 1))
        out = out.view(-1, 512) # (shape: (batch_size, 512))

        out = F.relu(self.fc1(out)) # (shape: (batch_size, 256))
        out = F.relu(self.fc2(out)) # (shape: (batch_size, 128))

        out = self.fc3(out) # (shape: (batch_size, 3))

        return out

class BboxNet(nn.Module):
    def __init__(self, num_seg_points=512):
        super(BboxNet, self).__init__()

        self.NH = 4 # (number of angle bins)

        self.num_seg_points = num_seg_points

        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.max_pool = nn.MaxPool1d(num_seg_points)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + 3 + 2*self.NH)

    def forward(self, x):
        # (x has shape (batch_size, 3, num_seg_points))

        out = F.relu(self.conv1(x)) # (shape: (batch_size, 128, num_seg_points))
        out = F.relu(self.conv2(out)) # (shape: (batch_size, 128, num_seg_points))
        out = F.relu(self.conv3(out)) # (shape: (batch_size, 256, num_seg_points))
        out = F.relu(self.conv4(out)) # (shape: (batch_size, 512, num_seg_points))
        out = self.max_pool(out) # (shape: (batch_size, 512, 1))
        out = out.view(-1, 512) # (shape: (batch_size, 512))

        out = F.relu(self.fc1(out)) # (shape: (batch_size, 512))
        out = F.relu(self.fc2(out)) # (shape: (batch_size, 256))

        out = self.fc3(out) # (shape: (batch_size, 3 + 3 + 2*NH))

        return out

class FrustumPointNet(nn.Module):
    def __init__(self, model_id, project_dir, num_points=1024):
        super(FrustumPointNet, self).__init__()

        self.num_points = num_points
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.InstanceSeg_network = InstanceSeg()
        self.TNet_network = TNet()
        self.BboxNet_network = BboxNet()

    def forward(self, x):
        batch_size = x.size()[0] # (x has shape (batch_size, 4, num_points))

        out_InstanceSeg = self.InstanceSeg_network(x) # (shape (batch_size, num_points, 2))

        ########################################################################
        point_clouds = x.transpose(2, 1)[:, :, 0:3].data.cpu().numpy() # (shape: (batch_size, num_points, 3))
        seg_scores = out_InstanceSeg.data.cpu().numpy() # (shape: (batch_size, num_points_ 2))

        seg_point_clouds = np.zeros((0, 512, 3), dtype=np.float32) # (shape: (batch_size, 512=num_seg_points, 3))
        out_dont_care_mask = torch.ones((batch_size, )) # (shape: (batch_size, ))
        out_dont_care_mask = out_dont_care_mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
        for i in range(seg_scores.shape[0]):
            ex_seg_scores = seg_scores[i] # (shape: (num_points, 2))
            ex_point_cloud = point_clouds[i] # (shape: (num_points, 3))

            row_mask = ex_seg_scores[:, 1] > ex_seg_scores[:, 0]
            ex_seg_point_cloud = ex_point_cloud[row_mask, :]

            if ex_seg_point_cloud.shape[0] == 0: # (if the segmented point cloud is empty:)
                ex_seg_point_cloud = np.zeros((512, 3), dtype=np.float32)
                out_dont_care_mask[i] = 0

            # randomly sample 512 points in ex_seg_point_cloud:
            if ex_seg_point_cloud.shape[0] < 512:
                row_idx = np.random.choice(ex_seg_point_cloud.shape[0], 512, replace=True)
            else:
                row_idx = np.random.choice(ex_seg_point_cloud.shape[0], 512, replace=False)
            ex_seg_point_cloud = ex_seg_point_cloud[row_idx, :]

            seg_point_clouds = np.concatenate((seg_point_clouds, [ex_seg_point_cloud]), axis=0)

        # subtract the point cloud centroid from each seg_point_cloud (transform to local coords):
        seg_point_clouds_mean = np.mean(seg_point_clouds, axis=1) # (shape: (batch_size, 3)) (seg_point_clouds has shape (batch_size, num_seg_points, 3))
        out_seg_point_clouds_mean = Variable(torch.from_numpy(seg_point_clouds_mean)).cuda()
        seg_point_clouds_mean = np.expand_dims(seg_point_clouds_mean, axis=1) # (shape: (batch_size, 1, 3))
        seg_point_clouds = seg_point_clouds - seg_point_clouds_mean

        seg_point_clouds = Variable(torch.from_numpy(seg_point_clouds)).cuda() # (shape: (batch_size, num_seg_points, 3))
        seg_point_clouds = seg_point_clouds.transpose(2, 1) # (shape: (batch_size, 3, num_seg_points))
        ########################################################################

        out_TNet = self.TNet_network(seg_point_clouds) # (shape: (batch_size, 3))

        # subtract the TNet predicted translation from each seg_point_cloud (transfrom to local coords):
        seg_point_clouds = seg_point_clouds - out_TNet.unsqueeze(2).repeat(1, 1, seg_point_clouds.size()[2]) # (out_TNet.unsqueeze(2).repeat(1, 1, seg_point_clouds.size()[2]) has shape: (batch_size, 3, num_seg_points))

        out_BboxNet = self.BboxNet_network(seg_point_clouds) # (shape: (batch_size, 3 + 3 + 2*NH))

        # (out_InstanceSeg has shape: (batch_size, num_points, 2))
        # (out_seg_point_clouds_mean has shape: (batch_size, 3))
        # (out_dont_care_mask has shape: (batch_size, ))

        return (out_InstanceSeg, out_TNet, out_BboxNet, out_seg_point_clouds_mean, out_dont_care_mask)

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

# from torch.autograd import Variable
# x = Variable(torch.randn(32, 3, 512))
# network = BboxNet()
# out = network(x)

# x = Variable(torch.randn(32, 4, 1024)).cuda()
# network = FrustumPointNet("frustum_pointnet_test", "/staging/frexgus/frustum_pointnet")
# network = network.cuda()
# out = network(x)
