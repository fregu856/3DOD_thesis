# 3DOD_thesis

NOTE! The uploaded code is NOT camera-ready yet, a final version will be released before the end of September.

- Youtube video of results (https://youtu.be/KdrHLXpYYlg):
[![demo video with results](https://img.youtube.com/vi/KdrHLXpYYlg/0.jpg)](https://www.youtube.com/watch?v=KdrHLXpYYlg)

******
## Paperspace:

To train models and to run pretrained models, you can use an Ubuntu 16.04 P4000 VM with 250 GB SSD on Paperspace. Below I have listed what I needed to do in order to get started, and some things I found useful.

- Install docker-ce:
- - $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
- - $ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
- - $ sudo apt-get update
- - $ sudo apt-get install -y docker-ce

- Install CUDA drivers:
- - $ CUDA_REPO_PKG=cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
- - $ wget -O /tmp/${CUDA_REPO_PKG} http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}
- - $ sudo dpkg -i /tmp/${CUDA_REPO_PKG}
- - $ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
- - $ rm -f /tmp/${CUDA_REPO_PKG}
- - $ sudo apt-get update
- - $ sudo apt-get install cuda-drivers
- - Reboot the VM.

- Install nvidia-docker:
- - $ wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
- - $ sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
- - $ sudo nvidia-docker run --rm nvidia/cuda nvidia-smi

- Download the PyTorch 0.4 docker image:
- - $ sudo docker pull pytorch/pytorch:0.4_cuda9_cudnn7

- Create start_docker_image.sh containing:
```
#!/bin/bash

# DEFAULT VALUES
GPUIDS="0"
NAME="paperspace_GPU"


NV_GPU="$GPUIDS" nvidia-docker run -it --rm \
        -p 5584:5584 \
        --name "$NAME""$GPUIDS" \
        -v /home/paperspace:/root/ \
        pytorch/pytorch:0.4_cuda9_cudnn7 bash
```

- Inside the image, /root/ will now be mapped to /home/paperspace (i.e., $ cd -- takes you to the regular home folder). 

- To start the image:
- - $ sudo sh start_docker_image.sh 
- To commit changes to the image:
- - Open a new terminal window.
- - $ sudo docker commit paperspace_GPU0 pytorch/pytorch:0.4_cuda9_cudnn7
- To stop the image when it’s running:
- - $ sudo docker stop paperspace_GPU0
- To exit the image without killing running code:
- - Ctrl + P + Q
- To get back into a running image:
- - $ sudo docker attach paperspace_GPU0
- To open more than one terminal window at the same time:
- - $ sudo docker exec -it paperspace_GPU0 bash

- To install the needed software inside the docker image:
- - $ apt-get update
- - $ apt-get install nano
- - $ apt-get install sudo
- - $ apt-get install wget
- - $ sudo apt install unzip
- - $ sudo apt-get install libopencv-dev
- - $ pip install opencv-python
- - $ python -mpip install matplotlib
- - Commit changes to the image (otherwise, the installed packages will be removed at exit!)

- Do the following outside of the docker image:
- - $ --

- - $ git clone https://github.com/fregu856/3DOD_thesis.git 

- - Download KITTI object (data_object_image_2.zip, data_object_velodyne.zip, data_object_calib.zip and data_object_label_2.zip) ($ wget *the download link that was sent to you in an email*).
- - Unzip all files ($ sudo apt install unzip, and then $ unzip *file name*).
- - Place the folders 'training' and 'testing' in 3DOD_thesis/data/kitti/object.

- - Download KITTI tracking (data_tracking_image_2.zip, data_tracking_velodyne.zip, data_tracking_calib.zip and data_tracking_label_2.zip) ($ wget *the download link that was sent to you in an email*).
- - Unzip all files ($ sudo apt install unzip, and then $ unzip *file name*).
- - Place the folders 'training' and 'testing' in 3DOD_thesis/data/kitti/tracking.

***
### Used datasets:
- *KITTI train*:
- - 3712 examples (roughly 50%) from KITTI training, see thesis for more info.

- *KITTI val*:
- - 3769 examples (roughly 50%) from KITTI training, see thesis for more info.

- *KITTI train random*:
- - 6733 examples (random 90% subset) from KITTI training.

- *KITTI test*:
- - The KITTI testing set, 7518 examples.

***
### Pretrained models:
- pretrained_models/model_37_2_epoch_400.pth:
- - Frustum-PointNet trained for 400 epochs on *KITTI train random*.

- pretrained_models/model_32_2_epoch_299.pth:
- - Frustum-PointNet trained for 299 epochs on *SYN train* (synthetic dataset, see the thesis for more info).

- pretrained_models/model_38_2_epoch_400.pth:
- - Extended-Frustum-PointNet trained for 400 epochs on *KITTI train random*.

- pretrained_models/model_10_2_epoch_400.pth:
- - Image-Only trained for 400 epochs on *KITTI train random*.

***
### Train Frustum-PointNet:
- SSH into the paperspace server.
- $ sudo sh start_docker_image.sh
- $ cd --
- $ python 3DOD_thesis/Frustum-PointNet/train_frustum_pointnet.py

***
### Train Extended-Frustum-PointNet:
- SSH into the paperspace server.
- $ sudo sh start_docker_image.sh
- $ cd --
- $ python 3DOD_thesis/Extended-Frustum-PointNet/train_frustum_pointnet_img.py

***
### Train Image-Only:
- SSH into the paperspace server.
- $ sudo sh start_docker_image.sh
- $ cd --
- $ python 3DOD_thesis/Image-Only/train_imgnet.py


***
#### :
- SSH into the paperspace server.
- $ sudo sh start_docker_image.sh
- $ cd --
- $ python 3DOD_thesis/Image-Only/train_imgnet.py
