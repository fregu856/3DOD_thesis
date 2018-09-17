# 3DOD_thesis

Code will be released before the end of September.

- Youtube video of results (https://youtu.be/KdrHLXpYYlg):
[![demo video with results](https://img.youtube.com/vi/KdrHLXpYYlg/0.jpg)](https://www.youtube.com/watch?v=KdrHLXpYYlg)

******
## Training on Microsoft Azure:

To train the model, I used an NC6 virtual machine on Microsoft Azure. Below I have listed what I needed to do in order to get started, and some things I found useful. For reference, my username was 'fregu856':



- Create start_docker_image.sh containing:
```
#!/bin/bash

# DEFAULT VALUES
GPUIDS="0"
NAME="fregu856_GPU"


NV_GPU="$GPUIDS" nvidia-docker run -it --rm \
        -p 5584:5584 \
        --name "$NAME""$GPUIDS" \
        -v /home/fregu856:/root/ \
        tensorflow/tensorflow:latest-gpu bash
```

- /root/ will now be mapped to /home/fregu856 (i.e., $ cd -- takes you to the regular home folder). 

- To start the image:
- - $ sudo sh start_docker_image.sh 
- To commit changes to the image:
- - Open a new terminal window.
- - $ sudo docker commit fregu856_GPU0 tensorflow/tensorflow:latest-gpu
- To stop the image when it’s running:
- - $ sudo docker stop fregu856_GPU0
- To exit the image without killing running code:
- - Ctrl-P + Q
- To get back into a running image:
- - $ sudo docker attach fregu856_GPU0
- To open more than one terminal window at the same time:
- - $ sudo docker exec -it fregu856_GPU0 bash

- To install the needed software inside the docker image:
- - $ apt-get update
- - $ apt-get install nano
- - $ apt-get install sudo
- - $ apt-get install wget
- - $ sudo apt-get install libopencv-dev python-opencv
- - Commit changes to the image (otherwise, the installed packages will be removed at exit!)
