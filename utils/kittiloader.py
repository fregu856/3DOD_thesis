# camera-ready

import numpy as np
import cv2
import math

################################################################################
# imported functions:
################################################################################
def calibread(file_path):
    out = dict()
    for line in open(file_path, 'r'):
        line = line.strip()
        if line == '' or line[0] == '#':
            continue
        val = line.split(':')
        assert len(val) == 2, 'Wrong file format, only one : per line!'
        key_name = val[0].strip()
        val = np.asarray(val[-1].strip().split(' '), dtype='f8')
        assert len(val) in [12, 9], "Wrong file format, wrong number of numbers!"
        if len(val) == 12:
            out[key_name] = val.reshape(3, 4)
        elif len(val) == 9:
            out[key_name] = val.reshape(3, 3)
    return out

def LabelLoader2D3D(file_id, path, ext, calib_path, calib_ext):
    labels = labelread(path + "/" + file_id + ext)
    calib = calibread(calib_path + "/" + file_id + calib_ext)
    polys = list()
    for bbox in labels:
        poly = dict()

        poly2d = dict()
        poly2d['class'] = bbox['type']
        poly2d['truncated'] = bbox['truncated']
        poly2d['poly'] = np.array([[bbox['bbox']['left'], bbox['bbox']['top']],
                                 [bbox['bbox']['right'], bbox['bbox']['top']],
                                 [bbox['bbox']['right'], bbox['bbox']['bottom']],
                                 [bbox['bbox']['left'], bbox['bbox']['bottom']]],
                                dtype='int32')
        poly["label_2D"] = poly2d

        poly3d = dict()
        poly3d['class'] = bbox['type']
        location = np.asarray([bbox['location']['x'],
                               bbox['location']['y'],
                               bbox['location']['z']], dtype='float32')
        r_y = bbox['rotation_y']
        Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)], [0, 1, 0],
                           [-math.sin(r_y), 0, math.cos(r_y)]],
                          dtype='float32')
        length = bbox['dimensions']['length']
        width = bbox['dimensions']['width']
        height = bbox['dimensions']['height']
        p0 = np.dot(Rmat, np.asarray(
            [length / 2.0, 0, width / 2.0], dtype='float32'))
        p1 = np.dot(Rmat, np.asarray(
            [-length / 2.0, 0, width / 2.0], dtype='float32'))
        p2 = np.dot(Rmat, np.asarray(
            [-length / 2.0, 0, -width / 2.0], dtype='float32'))
        p3 = np.dot(Rmat, np.asarray(
            [length / 2.0, 0, -width / 2.0], dtype='float32'))
        p4 = np.dot(Rmat, np.asarray(
            [length / 2.0, -height, width / 2.0], dtype='float32'))
        p5 = np.dot(Rmat, np.asarray(
            [-length / 2.0, -height, width / 2.0], dtype='float32'))
        p6 = np.dot(Rmat, np.asarray(
            [-length / 2.0, -height, -width / 2.0], dtype='float32'))
        p7 = np.dot(Rmat, np.asarray(
            [length / 2.0, -height, -width / 2.0], dtype='float32'))
        poly3d['points'] = np.array(location + [p0, p1, p2, p3, p4, p5, p6, p7])
        poly3d['lines'] = [[0, 3, 7, 4, 0], [1, 2, 6, 5, 1],
                         [0, 1], [2, 3], [6, 7], [4, 5]]
        poly3d['colors'] = [[255, 0, 0], [0, 0, 255], [
            255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]]
        poly3d['P0_mat'] = calib['P2']
        poly3d['center'] = location
        poly3d['l'] = length
        poly3d['w'] = width
        poly3d['h'] = height
        poly3d['r_y'] = r_y
        poly["label_3D"] = poly3d

        polys.append(poly)
    return polys

def LabelLoader2D3D_sequence(img_id, img_id_float, label_path, calib_path):
    labels = labelread_sequence(label_path)

    img_id_labels = []
    for label in labels:
        if label["frame"] == img_id_float:
            img_id_labels.append(label)

    calib = calibread(calib_path)
    polys = list()
    for bbox in img_id_labels:
        poly = dict()

        poly2d = dict()
        poly2d['class'] = bbox['type']
        poly2d['truncated'] = bbox['truncated']
        poly2d['occluded'] = bbox['occluded']
        poly2d['poly'] = np.array([[bbox['bbox']['left'], bbox['bbox']['top']],
                                 [bbox['bbox']['right'], bbox['bbox']['top']],
                                 [bbox['bbox']['right'], bbox['bbox']['bottom']],
                                 [bbox['bbox']['left'], bbox['bbox']['bottom']]],
                                dtype='int32')
        poly["label_2D"] = poly2d

        poly3d = dict()
        poly3d['class'] = bbox['type']
        location = np.asarray([bbox['location']['x'],
                               bbox['location']['y'],
                               bbox['location']['z']], dtype='float32')
        r_y = bbox['rotation_y']
        Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)], [0, 1, 0],
                           [-math.sin(r_y), 0, math.cos(r_y)]],
                          dtype='float32')
        length = bbox['dimensions']['length']
        width = bbox['dimensions']['width']
        height = bbox['dimensions']['height']
        p0 = np.dot(Rmat, np.asarray(
            [length / 2.0, 0, width / 2.0], dtype='float32'))
        p1 = np.dot(Rmat, np.asarray(
            [-length / 2.0, 0, width / 2.0], dtype='float32'))
        p2 = np.dot(Rmat, np.asarray(
            [-length / 2.0, 0, -width / 2.0], dtype='float32'))
        p3 = np.dot(Rmat, np.asarray(
            [length / 2.0, 0, -width / 2.0], dtype='float32'))
        p4 = np.dot(Rmat, np.asarray(
            [length / 2.0, -height, width / 2.0], dtype='float32'))
        p5 = np.dot(Rmat, np.asarray(
            [-length / 2.0, -height, width / 2.0], dtype='float32'))
        p6 = np.dot(Rmat, np.asarray(
            [-length / 2.0, -height, -width / 2.0], dtype='float32'))
        p7 = np.dot(Rmat, np.asarray(
            [length / 2.0, -height, -width / 2.0], dtype='float32'))
        poly3d['points'] = np.array(location + [p0, p1, p2, p3, p4, p5, p6, p7])
        poly3d['lines'] = [[0, 3, 7, 4, 0], [1, 2, 6, 5, 1],
                         [0, 1], [2, 3], [6, 7], [4, 5]]
        poly3d['colors'] = [[255, 0, 0], [0, 0, 255], [
            255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]]
        poly3d['P0_mat'] = calib['P2']
        poly3d['center'] = location
        poly3d['l'] = length
        poly3d['w'] = width
        poly3d['h'] = height
        poly3d['r_y'] = r_y
        poly["label_3D"] = poly3d

        polys.append(poly)
    return polys

def labelread(file_path):
    bbox = ('bbox', ['left', 'top', 'right', 'bottom'])
    dimensions = ('dimensions', ['height', 'width', 'length'])
    location = ('location', ['x', 'y', 'z'])
    keys = ['type', 'truncated', 'occluded', 'alpha', bbox,
            dimensions, location, 'rotation_y', 'score']
    labels = list()
    for line in open(file_path, 'r'):
        vals = line.split()
        l, _ = vals_to_dict(vals, keys)
        labels.append(l)
    return labels

################################################################################
# helper functions:
################################################################################
def labelread_sequence(file_path):
    bbox = ('bbox', ['left', 'top', 'right', 'bottom'])
    dimensions = ('dimensions', ['height', 'width', 'length'])
    location = ('location', ['x', 'y', 'z'])
    keys = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', bbox,
            dimensions, location, 'rotation_y', 'score']
    labels = list()
    for line in open(file_path, 'r'):
        vals = line.split()
        l, _ = vals_to_dict(vals, keys)
        labels.append(l)
    return labels

def vals_to_dict(vals, keys, vals_n=0):
    out = dict()
    for key in keys:
        if isinstance(key, str):
            try:
                val = float(vals[vals_n])
            except:
                val = vals[vals_n]
            data = val
            key_name = key
            vals_n += 1
        else:
            data, vals_n = vals_to_dict(vals, key[1], vals_n)
            key_name = key[0]
        out[key_name] = data
        if vals_n >= len(vals):
            break
    return out, vals_n
