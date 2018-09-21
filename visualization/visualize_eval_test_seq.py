# camera-ready

import pickle
import numpy as np
import math
import cv2
import os

import sys
sys.path.append("/home/fregu856/3DOD_thesis/Open3D/build/lib") # NOTE! you'll have to adapt this for your file structure
from py3d import *

sys.path.append("/home/fregu856/3DOD_thesis/utils") # NOTE! you'll have to adapt this for your file structure
from kittiloader import LabelLoader2D3D, calibread

def create3Dbbox(center, h, w, l, r_y, type="pred"):
    if type == "pred":
        color = [1, 0.75, 0] # (normalized RGB)
        front_color = [1, 0, 0] # (normalized RGB)
    else: # (if type == "gt":)
        color = [1, 0, 0.75] # (normalized RGB)
        front_color = [0, 0.9, 1] # (normalized RGB)

    Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)],
                       [0, 1, 0],
                       [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype='float32')

    Rmat_90 = np.asarray([[math.cos(r_y+np.pi/2), 0, math.sin(r_y+np.pi/2)],
                          [0, 1, 0],
                          [-math.sin(r_y+np.pi/2), 0, math.cos(r_y+np.pi/2)]],
                          dtype='float32')

    Rmat_90_x = np.asarray([[1, 0, 0],
                            [0, math.cos(np.pi/2), math.sin(np.pi/2)],
                            [0, -math.sin(np.pi/2), math.cos(np.pi/2)]],
                            dtype='float32')

    p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())

    p0_3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, 0], dtype='float32').flatten())
    p1_2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, 0], dtype='float32').flatten())
    p4_7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, 0], dtype='float32').flatten())
    p5_6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, 0], dtype='float32').flatten())
    p0_1 = center + np.dot(Rmat, np.asarray([0, 0, w/2.0], dtype='float32').flatten())
    p3_2 = center + np.dot(Rmat, np.asarray([0, 0, -w/2.0], dtype='float32').flatten())
    p4_5 = center + np.dot(Rmat, np.asarray([0, -h, w/2.0], dtype='float32').flatten())
    p7_6 = center + np.dot(Rmat, np.asarray([0, -h, -w/2.0], dtype='float32').flatten())
    p0_4 = center + np.dot(Rmat, np.asarray([l/2.0, -h/2.0, w/2.0], dtype='float32').flatten())
    p3_7 = center + np.dot(Rmat, np.asarray([l/2.0, -h/2.0, -w/2.0], dtype='float32').flatten())
    p1_5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h/2.0, w/2.0], dtype='float32').flatten())
    p2_6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h/2.0, -w/2.0], dtype='float32').flatten())
    p0_1_3_2 = center

    length_0_3 = np.linalg.norm(p0 - p3)
    cylinder_0_3 = create_mesh_cylinder(radius=0.035, height=length_0_3)
    cylinder_0_3.compute_vertex_normals()
    transform_0_3 = np.eye(4)
    transform_0_3[0:3, 0:3] = Rmat
    transform_0_3[0:3, 3] = p0_3
    cylinder_0_3.transform(transform_0_3)
    cylinder_0_3.paint_uniform_color(front_color)

    length_1_2 = np.linalg.norm(p1 - p2)
    cylinder_1_2 = create_mesh_cylinder(radius=0.035, height=length_1_2)
    cylinder_1_2.compute_vertex_normals()
    transform_1_2 = np.eye(4)
    transform_1_2[0:3, 0:3] = Rmat
    transform_1_2[0:3, 3] = p1_2
    cylinder_1_2.transform(transform_1_2)
    cylinder_1_2.paint_uniform_color(color)

    length_4_7 = np.linalg.norm(p4 - p7)
    cylinder_4_7 = create_mesh_cylinder(radius=0.035, height=length_4_7)
    cylinder_4_7.compute_vertex_normals()
    transform_4_7 = np.eye(4)
    transform_4_7[0:3, 0:3] = Rmat
    transform_4_7[0:3, 3] = p4_7
    cylinder_4_7.transform(transform_4_7)
    cylinder_4_7.paint_uniform_color(front_color)

    length_5_6 = np.linalg.norm(p5 - p6)
    cylinder_5_6 = create_mesh_cylinder(radius=0.035, height=length_5_6)
    cylinder_5_6.compute_vertex_normals()
    transform_5_6 = np.eye(4)
    transform_5_6[0:3, 0:3] = Rmat
    transform_5_6[0:3, 3] = p5_6
    cylinder_5_6.transform(transform_5_6)
    cylinder_5_6.paint_uniform_color(color)

    # #

    length_0_1 = np.linalg.norm(p0 - p1)
    cylinder_0_1 = create_mesh_cylinder(radius=0.035, height=length_0_1)
    cylinder_0_1.compute_vertex_normals()
    transform_0_1 = np.eye(4)
    transform_0_1[0:3, 0:3] = Rmat_90
    transform_0_1[0:3, 3] = p0_1
    cylinder_0_1.transform(transform_0_1)
    cylinder_0_1.paint_uniform_color(color)

    length_3_2 = np.linalg.norm(p3 - p2)
    cylinder_3_2 = create_mesh_cylinder(radius=0.035, height=length_3_2)
    cylinder_3_2.compute_vertex_normals()
    transform_3_2 = np.eye(4)
    transform_3_2[0:3, 0:3] = Rmat_90
    transform_3_2[0:3, 3] = p3_2
    cylinder_3_2.transform(transform_3_2)
    cylinder_3_2.paint_uniform_color(color)

    length_4_5 = np.linalg.norm(p4 - p5)
    cylinder_4_5 = create_mesh_cylinder(radius=0.035, height=length_4_5)
    cylinder_4_5.compute_vertex_normals()
    transform_4_5 = np.eye(4)
    transform_4_5[0:3, 0:3] = Rmat_90
    transform_4_5[0:3, 3] = p4_5
    cylinder_4_5.transform(transform_4_5)
    cylinder_4_5.paint_uniform_color(color)

    length_7_6 = np.linalg.norm(p7 - p6)
    cylinder_7_6 = create_mesh_cylinder(radius=0.035, height=length_7_6)
    cylinder_7_6.compute_vertex_normals()
    transform_7_6 = np.eye(4)
    transform_7_6[0:3, 0:3] = Rmat_90
    transform_7_6[0:3, 3] = p7_6
    cylinder_7_6.transform(transform_7_6)
    cylinder_7_6.paint_uniform_color(color)

    # #

    length_0_4 = np.linalg.norm(p0 - p4)
    cylinder_0_4 = create_mesh_cylinder(radius=0.035, height=length_0_4)
    cylinder_0_4.compute_vertex_normals()
    transform_0_4 = np.eye(4)
    transform_0_4[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_0_4[0:3, 3] = p0_4
    cylinder_0_4.transform(transform_0_4)
    cylinder_0_4.paint_uniform_color(front_color)

    length_3_7 = np.linalg.norm(p3 - p7)
    cylinder_3_7 = create_mesh_cylinder(radius=0.035, height=length_3_7)
    cylinder_3_7.compute_vertex_normals()
    transform_3_7 = np.eye(4)
    transform_3_7[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_3_7[0:3, 3] = p3_7
    cylinder_3_7.transform(transform_3_7)
    cylinder_3_7.paint_uniform_color(front_color)

    length_1_5 = np.linalg.norm(p1 - p5)
    cylinder_1_5 = create_mesh_cylinder(radius=0.035, height=length_1_5)
    cylinder_1_5.compute_vertex_normals()
    transform_1_5 = np.eye(4)
    transform_1_5[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_1_5[0:3, 3] = p1_5
    cylinder_1_5.transform(transform_1_5)
    cylinder_1_5.paint_uniform_color(color)

    length_2_6 = np.linalg.norm(p2 - p6)
    cylinder_2_6 = create_mesh_cylinder(radius=0.035, height=length_2_6)
    cylinder_2_6.compute_vertex_normals()
    transform_2_6 = np.eye(4)
    transform_2_6[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_2_6[0:3, 3] = p2_6
    cylinder_2_6.transform(transform_2_6)
    cylinder_2_6.paint_uniform_color(color)

    # #

    length_0_1_3_2 = np.linalg.norm(p0_1 - p3_2)
    cylinder_0_1_3_2 = create_mesh_cylinder(radius=0.035, height=length_0_1_3_2)
    cylinder_0_1_3_2.compute_vertex_normals()
    transform_0_1_3_2 = np.eye(4)
    transform_0_1_3_2[0:3, 0:3] = Rmat
    transform_0_1_3_2[0:3, 3] = p0_1_3_2
    cylinder_0_1_3_2.transform(transform_0_1_3_2)
    cylinder_0_1_3_2.paint_uniform_color(color)

    return [cylinder_0_1_3_2, cylinder_0_3, cylinder_1_2, cylinder_4_7, cylinder_5_6, cylinder_0_1, cylinder_3_2, cylinder_4_5, cylinder_7_6, cylinder_0_4, cylinder_3_7, cylinder_1_5, cylinder_2_6]

def create3Dbbox_poly(center, h, w, l, r_y, P2_mat, type="pred"):
    if type == "pred":
        color = [0, 190, 255] # (BGR)
        front_color = [0, 0, 255] # (BGR)
    else: # (if type == "gt":)
        color = [190, 0, 255] # (BGR)
        front_color = [255, 230, 0] # (BGR)

    poly = {}

    Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)],
                       [0, 1, 0],
                       [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype='float32')

    p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())

    poly['points'] = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
    poly['lines'] = [[0, 3, 7, 4, 0], [1, 2, 6, 5, 1], [0, 1], [2, 3], [6, 7], [4, 5]] # (0 -> 3 -> 7 -> 4 -> 0, 1 -> 2 -> 6 -> 5 -> 1, etc.)
    poly['colors'] = [front_color, color, color, color, color, color]
    poly['P0_mat'] = P2_mat

    return poly

def create2Dbbox_poly(bbox2D):
    u_min = bbox2D[0] # (left)
    u_max = bbox2D[1] # (rigth)
    v_min = bbox2D[2] # (top)
    v_max = bbox2D[3] # (bottom)

    poly = {}
    poly['poly'] = np.array([[u_min, v_min], [u_max, v_min],
                             [u_max, v_max], [u_min, v_max]], dtype='int32')

    return poly

def draw_2d_polys_no_text(img, polys):
    img = np.copy(img)
    for poly in polys:
        if 'color' in poly:
            bg = poly['color']
        else:
            bg = np.array([0, 255, 0], dtype='float64')

        cv2.polylines(img, np.int32([poly['poly']]), True, bg, lineType=cv2.LINE_AA, thickness=2)

    return img

def draw_3d_polys(img, polys):
    img = np.copy(img)
    for poly in polys:
        for n, line in enumerate(poly['lines']):
            if 'colors' in poly:
                bg = poly['colors'][n]
            else:
                bg = np.array([255, 0, 0], dtype='float64')

            p3d = np.vstack((poly['points'][line].T, np.ones((1, poly['points'][line].shape[0]))))
            p2d = np.dot(poly['P0_mat'], p3d)

            for m, p in enumerate(p2d[2, :]):
                p2d[:, m] = p2d[:, m]/p

            cv2.polylines(img, np.int32([p2d[:2, :].T]), False, bg, lineType=cv2.LINE_AA, thickness=2)

    return img

for sequence in ["0001", "0002", "0007", "0011"]:
    print (sequence)

    project_dir = "/home/fregu856/3DOD_thesis/" # NOTE! you'll have to adapt this for your file structure
    data_dir = project_dir + "data/kitti/tracking/testing/"
    img_dir = data_dir + "image_02/" + sequence + "/"
    calib_path = project_dir + "data/kitti/meta/tracking/testing/calib/" + sequence + ".txt" # NOTE! kitti/meta
    lidar_dir = data_dir + "velodyne/" + sequence + "/"

    calib = calibread(calib_path)
    P2 = calib["P2"]
    Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
    R0_rect_orig = calib["R0_rect"]

    R0_rect = np.eye(4)
    R0_rect[0:3, 0:3] = R0_rect_orig

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

    # NOTE! here you can choose what model's output you want to visualize
    # Frustum-PointNet:
    with open("/home/fregu856/3DOD_thesis/training_logs/model_Frustum-PointNet_eval_test_seq/eval_dict_test_seq_%s.pkl" % sequence, "rb") as file: # NOTE! you'll have to adapt this for your file structure
        eval_dict = pickle.load(file)
    #################################
    # # Extended-Frustum-PointNet:
    # with open("/home/fregu856/3DOD_thesis/training_logs/model_Extended-Frustum-PointNet_eval_test_seq/eval_dict_test_seq_%s.pkl" % sequence, "rb") as file: # NOTE! you'll have to adapt this for your file structure
    #     eval_dict = pickle.load(file)
    # #################################
    # # Image-Only:
    # with open("/home/fregu856/3DOD_thesis/training_logs/model_Image-Only_eval_test_seq/eval_dict_test_seq_%s.pkl" % sequence, "rb") as file: # NOTE! you'll have to adapt this for your file structure
    #     eval_dict = pickle.load(file)

    img_data_dict = {}
    for img_id in eval_dict:
        data_dict = {}

        bbox_dicts = eval_dict[img_id]

        pred_bboxes = []
        pred_bbox_polys = []
        input_2Dbbox_polys = []
        pred_seg_pcds = []
        for bbox_dict in bbox_dicts:
            # frustum_point_cloud = bbox_dict["frustum_point_cloud"]
            # pred_seg_point_cloud = bbox_dict["pred_seg_point_cloud"]
            input_2Dbbox = bbox_dict["input_2Dbbox"]
            pred_center_BboxNet = bbox_dict["pred_center_BboxNet"]
            pred_h = bbox_dict["pred_h"]
            pred_w = bbox_dict["pred_w"]
            pred_l = bbox_dict["pred_l"]
            pred_r_y = bbox_dict["pred_r_y"]

            # pred_seg_pcd = PointCloud()
            # pred_seg_pcd.points = Vector3dVector(pred_seg_point_cloud[:, 0:3])
            # pred_seg_pcd.paint_uniform_color([1, 0, 0])
            # pred_seg_pcds.append(pred_seg_pcd)

            input_2Dbbox_poly = create2Dbbox_poly(input_2Dbbox)
            input_2Dbbox_polys.append(input_2Dbbox_poly)

            pred_bbox_poly = create3Dbbox_poly(pred_center_BboxNet, pred_h, pred_w, pred_l, pred_r_y, P2, type="pred")
            pred_bbox_polys.append(pred_bbox_poly)

            pred_bbox = create3Dbbox(pred_center_BboxNet, pred_h, pred_w, pred_l, pred_r_y, type="pred")
            pred_bboxes += pred_bbox

        data_dict["pred_bboxes"] = pred_bboxes
        data_dict["pred_bbox_polys"] = pred_bbox_polys
        data_dict["input_2Dbbox_polys"] = input_2Dbbox_polys
        data_dict["pred_seg_pcds"] = pred_seg_pcds

        img_data_dict[img_id] = data_dict

    sorted_img_ids = []
    img_names = sorted(os.listdir(img_dir))
    for img_name in img_names:
        img_id = img_name.split(".png")[0]
        sorted_img_ids.append(img_id)

    img_height = 375
    img_width = 1242

    small_img_height = 187
    small_img_width = 620

    # ################################################################################
    # # create a video of images (no bboxes):
    # ################################################################################
    # out = cv2.VideoWriter("eval_test_seq_%s_img.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (img_width, img_height), True)
    #
    # for img_id in sorted_img_ids:
    #     print img_id
    #
    #     img = cv2.imread(img_dir + img_id + ".png", -1)
    #
    #     img = cv2.resize(img, (img_width, img_height)) # (the image MUST have the size specified in VideoWriter)
    #
    #     out.write(img)

    # ################################################################################
    # # create a video of images with pred:
    # ################################################################################
    # out = cv2.VideoWriter("eval_test_seq_%s_img_pred.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (img_width, img_height), True)
    #
    # for img_id in sorted_img_ids:
    #     print img_id
    #
    #     img = cv2.imread(img_dir + img_id + ".png", -1)
    #
    #     img_with_pred_bboxes = img
    #
    #     if img_id in img_data_dict:
    #         data_dict = img_data_dict[img_id]
    #         pred_bbox_polys = data_dict["pred_bbox_polys"]
    #
    #         img_with_pred_bboxes = draw_3d_polys(img, pred_bbox_polys)
    #
    #     img_with_pred_bboxes = cv2.resize(img_with_pred_bboxes, (img_width, img_height)) # (the image MUST have the size specified in VideoWriter)
    #
    #     out.write(img_with_pred_bboxes)

    # ################################################################################
    # # create a video of images with input 2Dbboxes on top of pred 3Dbboxes:
    # ################################################################################
    # out = cv2.VideoWriter("eval_test_seq_%s_img_input_pred.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (img_width, 2*img_height), True)
    #
    # for img_id in sorted_img_ids:
    #     print img_id
    #
    #     img = cv2.imread(img_dir + img_id + ".png", -1)
    #
    #     img_with_input_2Dbboxes = img
    #     img_with_pred_bboxes = img
    #
    #     if img_id in img_data_dict:
    #         data_dict = img_data_dict[img_id]
    #         input_2Dbbox_polys = data_dict["input_2Dbbox_polys"]
    #         pred_bbox_polys = data_dict["pred_bbox_polys"]
    #
    #         img_with_input_2Dbboxes = draw_2d_polys_no_text(img, input_2Dbbox_polys)
    #         img_with_pred_bboxes = draw_3d_polys(img, pred_bbox_polys)
    #
    #     img_with_input_2Dbboxes = cv2.resize(img_with_input_2Dbboxes, (img_width, img_height)) # (the image MUST have the size specified in VideoWriter)
    #     img_with_pred_bboxes = cv2.resize(img_with_pred_bboxes, (img_width, img_height)) # (the image MUST have the size specified in VideoWriter)
    #
    #     combined_img = np.zeros((2*img_height, img_width, 3), dtype=np.uint8)
    #     combined_img[0:img_height] = img_with_input_2Dbboxes
    #     combined_img[img_height:] = img_with_pred_bboxes
    #
    #     out.write(combined_img)

    class ImgCreatorLiDAR:
        def __init__(self):
            self.counter = 0
            self.trajectory = read_pinhole_camera_trajectory("/home/fregu856/3DOD_thesis/visualization/camera_trajectory.json") # NOTE! you'll have to adapt this for your file structure

        def move_forward(self, vis):
            # this function is called within the Visualizer::run() loop.
            # the run loop calls the function, then re-renders the image.

            if self.counter < 2: # (the counter is for making sure the camera view has been changed before the img is captured)
                # set the camera view:
                ctr = vis.get_view_control()
                ctr.convert_from_pinhole_camera_parameters(self.trajectory.intrinsic, self.trajectory.extrinsic[0])

                self.counter += 1
            else:
                # capture an image:
                img = vis.capture_screen_float_buffer()
                img = 255*np.asarray(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.uint8)
                self.lidar_img = img

                # close the window:
                vis.destroy_window()

                self.counter = 0

            return False

        def create_img(self, geometries):
            vis = Visualizer()
            vis.create_window()
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            for geometry in geometries:
                vis.add_geometry(geometry)
            vis.register_animation_callback(self.move_forward)
            vis.run()

            return self.lidar_img

    # ################################################################################
    # # create a video of LiDAR (no bboxes):
    # ################################################################################
    # out_lidar = cv2.VideoWriter("eval_test_seq_%s_lidar.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (1920, 1080), True)
    #
    # lidar_img_creator = ImgCreatorLiDAR()
    # for img_id in sorted_img_ids:
    #     print img_id
    #
    #     lidar_path = lidar_dir + img_id + ".bin"
    #     point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    #
    #     # remove points that are located behind the camera:
    #     point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]
    #
    #     point_cloud_xyz = point_cloud[:, 0:3]
    #     point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
    #     point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))
    #
    #     # transform the points into (rectified) camera coordinates:
    #     point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
    #     # normalize:
    #     point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
    #     point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]
    #
    #     pcd = PointCloud()
    #     pcd.points = Vector3dVector(point_cloud_xyz_camera)
    #     pcd.paint_uniform_color([0.65, 0.65, 0.65])
    #
    #     img = lidar_img_creator.create_img([pcd])
    #     out_lidar.write(img)

    # ################################################################################
    # # create a video of LiDAR with pred:
    # ################################################################################
    # out_lidar_pred = cv2.VideoWriter("eval_test_seq_%s_lidar_pred.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (1920, 1080), True)
    #
    # lidar_img_creator = ImgCreatorLiDAR()
    # for img_id in sorted_img_ids:
    #     print img_id
    #
    #     lidar_path = lidar_dir + img_id + ".bin"
    #     point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    #
    #     # remove points that are located behind the camera:
    #     point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]
    #
    #     point_cloud_xyz = point_cloud[:, 0:3]
    #     point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
    #     point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))
    #
    #     # transform the points into (rectified) camera coordinates:
    #     point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
    #     # normalize:
    #     point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
    #     point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]
    #
    #     pcd = PointCloud()
    #     pcd.points = Vector3dVector(point_cloud_xyz_camera)
    #     pcd.paint_uniform_color([0.65, 0.65, 0.65])
    #
    #     pred_bboxes = []
    #
    #     if img_id in img_data_dict:
    #         data_dict = img_data_dict[img_id]
    #         pred_bboxes = data_dict["pred_bboxes"]
    #
    #     img = lidar_img_creator.create_img(pred_bboxes + [pcd])
    #     out_lidar_pred.write(img)

    # ################################################################################
    # # create a video of image and LiDAR (no bboxes):
    # ################################################################################
    # out_lidar_img = cv2.VideoWriter("eval_test_seq_%s_lidar_img.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (1920, 1080), True)
    #
    # lidar_img_creator = ImgCreatorLiDAR()
    # for img_id in sorted_img_ids:
    #     print img_id
    #
    #     img = cv2.imread(img_dir + img_id + ".png", -1)
    #     small_img = cv2.resize(img, (small_img_width, small_img_height))
    #
    #     lidar_path = lidar_dir + img_id + ".bin"
    #     point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    #
    #     # remove points that are located behind the camera:
    #     point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]
    #
    #     point_cloud_xyz = point_cloud[:, 0:3]
    #     point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
    #     point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))
    #
    #     # transform the points into (rectified) camera coordinates:
    #     point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
    #     # normalize:
    #     point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
    #     point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]
    #
    #     pcd = PointCloud()
    #     pcd.points = Vector3dVector(point_cloud_xyz_camera)
    #     pcd.paint_uniform_color([0.65, 0.65, 0.65])
    #
    #     img_lidar = lidar_img_creator.create_img([pcd])
    #
    #     combined_img = img_lidar
    #     combined_img[-small_img_height:, ((1920/2)-(small_img_width/2)):((1920/2)+(small_img_width/2))] = small_img
    #
    #     out_lidar_img.write(combined_img)

    # ################################################################################
    # # create a video of image and LiDAR with pred:
    # ################################################################################
    # out_lidar_img_pred = cv2.VideoWriter("eval_test_seq_%s_lidar_img_pred.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (1920, 1080), True)
    #
    # lidar_img_creator = ImgCreatorLiDAR()
    # for img_id in sorted_img_ids:
    #     print img_id
    #
    #     img = cv2.imread(img_dir + img_id + ".png", -1)
    #     img_with_pred_bboxes = img
    #     if img_id in img_data_dict:
    #         data_dict = img_data_dict[img_id]
    #         pred_bbox_polys = data_dict["pred_bbox_polys"]
    #         img_with_pred_bboxes = draw_3d_polys(img, pred_bbox_polys)
    #     small_img_with_pred_bboxes = cv2.resize(img_with_pred_bboxes, (small_img_width, small_img_height))
    #
    #     lidar_path = lidar_dir + img_id + ".bin"
    #     point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    #
    #     # remove points that are located behind the camera:
    #     point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]
    #
    #     point_cloud_xyz = point_cloud[:, 0:3]
    #     point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
    #     point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))
    #
    #     # transform the points into (rectified) camera coordinates:
    #     point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
    #     # normalize:
    #     point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
    #     point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]
    #
    #     pcd = PointCloud()
    #     pcd.points = Vector3dVector(point_cloud_xyz_camera)
    #     pcd.paint_uniform_color([0.65, 0.65, 0.65])
    #
    #     pred_bboxes = []
    #
    #     if img_id in img_data_dict:
    #         data_dict = img_data_dict[img_id]
    #         pred_bboxes = data_dict["pred_bboxes"]
    #
    #     img_lidar = lidar_img_creator.create_img(pred_bboxes + [pcd])
    #
    #     combined_img = img_lidar
    #     combined_img[-small_img_height:, ((1920/2)-(small_img_width/2)):((1920/2)+(small_img_width/2))] = small_img_with_pred_bboxes
    #
    #     out_lidar_img_pred.write(combined_img)

    ################################################################################
    # create a video of image and LiDAR with input 2Dbboxes and pred 3Dbboxes:
    ################################################################################
    out_lidar_img_pred = cv2.VideoWriter("eval_test_seq_%s_lidar_img_input_pred.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (1920, 1080), True)

    lidar_img_creator = ImgCreatorLiDAR()
    for img_id in sorted_img_ids:
        print img_id

        img = cv2.imread(img_dir + img_id + ".png", -1)

        img_with_input_2Dbboxes = img
        img_with_pred_bboxes = img
        if img_id in img_data_dict:
            data_dict = img_data_dict[img_id]
            input_2Dbbox_polys = data_dict["input_2Dbbox_polys"]
            pred_bbox_polys = data_dict["pred_bbox_polys"]

            img_with_input_2Dbboxes = draw_2d_polys_no_text(img, input_2Dbbox_polys)
            img_with_pred_bboxes = draw_3d_polys(img, pred_bbox_polys)

        small_img_with_input_2Dbboxes = cv2.resize(img_with_input_2Dbboxes, (small_img_width, small_img_height))
        small_img_with_pred_bboxes = cv2.resize(img_with_pred_bboxes, (small_img_width, small_img_height))

        lidar_path = lidar_dir + img_id + ".bin"
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

        # remove points that are located behind the camera:
        point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]

        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

        # transform the points into (rectified) camera coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

        pcd = PointCloud()
        pcd.points = Vector3dVector(point_cloud_xyz_camera)
        pcd.paint_uniform_color([0.65, 0.65, 0.65])

        pred_bboxes = []

        if img_id in img_data_dict:
            data_dict = img_data_dict[img_id]
            pred_bboxes = data_dict["pred_bboxes"]

        img_lidar = lidar_img_creator.create_img(pred_bboxes + [pcd])

        combined_img = img_lidar
        combined_img[-small_img_height:, ((1920/2)-small_img_width-5):((1920/2)-5)] = small_img_with_input_2Dbboxes
        combined_img[-small_img_height:, ((1920/2)+5):((1920/2)+5+small_img_width)] = small_img_with_pred_bboxes

        out_lidar_img_pred.write(combined_img)
