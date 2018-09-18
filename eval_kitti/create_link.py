#!/usr/bin/env python

import sys
import os
import numpy as np

if len(sys.argv)<2:
    print 'Usage: parser.py results_folder database'

result_sha = sys.argv[1]
database = sys.argv[2]
LFRCNN = os.path.join(os.path.expanduser('~'), 'lsi-faster-rcnn')
output_folder = os.path.join('output', sys.argv[1], 'kitti_'+database)

i = 0
dirs = os.listdir(os.path.join(LFRCNN,output_folder))
for dir in dirs:
    print i, dir
    i+=1

var = raw_input("Please select. ")

iter_folder = os.path.join(LFRCNN, output_folder, dirs[int(var)])

i = 0
dirs = os.listdir(iter_folder)
for dir in dirs:
    print i, dir
    i+=1

var = raw_input("Please select. ")

txts_folder = os.path.join(iter_folder, dirs[int(var)])

print txts_folder

var = raw_input("Write name. ")

dst_folder = os.path.join('results',var)
dst = os.path.join(dst_folder,'data')

if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

os.symlink(txts_folder, dst)
print 'Created link from', txts_folder, 'to', os.path.join('results',var,'data')
