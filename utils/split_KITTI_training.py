# mostly donee

import os
import numpy as np
import cPickle

train_file_path = "/home/fregu856/exjobb/data/kitti/meta/train.txt"
val_file_path = "/home/fregu856/exjobb/data/kitti/meta/val.txt"

train_img_ids = []
with open(train_file_path) as train_file:
    for line in train_file:
        img_id = line.strip()
        train_img_ids.append(img_id)

val_img_ids = []
with open(val_file_path) as val_file:
    for line in val_file:
        img_id = line.strip()
        val_img_ids.append(img_id)

cPickle.dump(train_img_ids, open("/home/fregu856/exjobb/data/kitti/meta/train_img_ids.pkl", "w"))
cPickle.dump(val_img_ids, open("/home/fregu856/exjobb/data/kitti/meta/val_img_ids.pkl", "w"))

# train_img_ids = cPickle.load(open("/home/fregu856/exjobb/data/kitti/meta/train_img_ids.pkl"))
# val_img_ids = cPickle.load(open("/home/fregu856/exjobb/data/kitti/meta/val_img_ids.pkl"))
#
# print len(train_img_ids)
# print len(val_img_ids)
