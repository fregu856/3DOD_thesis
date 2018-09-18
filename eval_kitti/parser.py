#!/usr/bin/env python

import sys
import os
import numpy as np

# classes = ['car', 'pedestrian', 'cyclist', 'van', 'truck', 'person_sitting', 'tram']
classes = ['car']
difficulties = ['easy', 'moderate', 'hard']
params = ['detection', 'detection_ground', 'detection_3d']

if len(sys.argv)<2:
    print 'Usage: parser.py results_folder'

result_sha = sys.argv[1]
txt_dir = os.path.join('build','results', result_sha)

for class_name in classes:
    for param in params:
        txt_name = os.path.join(txt_dir, 'stats_' + class_name + '_' + param + '.txt')

        if not os.path.isfile(txt_name):
            print txt_name, 'not found'
            continue

        cont = np.loadtxt(txt_name)

        for idx, difficulty in enumerate(difficulties):
            sum = 0;
            for i in xrange(0, 41, 4): # NOTE! had to change from xrange(0, 40, 4)
                sum += cont[idx][i]

            average = sum/11.0
            print class_name, difficulty, param, average

        print '----------------'

    print '================='
