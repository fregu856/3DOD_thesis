# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

import os

class ImgNet(nn.Module):
    def __init__(self, model_id, project_dir):
        super(ImgNet, self).__init__()

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        resnet34 = models.resnet34()
        # load pretrained model:
        resnet34.load_state_dict(torch.load("/root/3DOD_thesis/pretrained_models/resnet/resnet34-333f7ec4.pth"))
        # remove fully connected layer:
        self.resnet34 = nn.Sequential(*list(resnet34.children())[:-2])

        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2*8 + 3 + 1)

    def forward(self, img):
        # (img has shape (batch_size, 3, H, W))

        out = self.resnet34(img) # (shape: (batch_size, 512, 7, 7))
        out = self.avg_pool(out) # (shape: (batch_size, 512, 1, 1))
        out = out.view(-1, 512) # (shape: (batch_size, 512))

        out = F.relu(self.fc1(out)) # (shape: (batch_size, 256))
        out = F.relu(self.fc2(out)) # (shape: (batch_size, 128)))

        out = self.fc3(out) # (shape: (batch_size, 2*8 + 3 + 1 = 20))

        return out

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

# x = Variable(torch.randn(32, 3, 224, 224)).cuda()
# network = ImgNet("ImgNet_test", "/staging/frexgus/imgnet")
# network = network.cuda()
# out = network(x)
