from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool
import os
import tensorflow as tf
import pandas as pd
import cv2
import PIL
from PIL import Image
import numpy as np
from tqdm import tqdm, tqdm_notebook
import argparse
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R
import io
from typing import Tuple, List
from PIL import Image

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

import utils.anchor_helper
from utils.frame_helper import FrameCalibrationData
import utils.frame_helper as frame_helper
import utils.preproc_helper as preproc_helper
import utils.bev_helper as bev_helper
import utils.rpn_helper as rpn_helper
import utils.rpn_keras as rpn_keras

gpus = tf.compat.v1.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1) 
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True, gpu_options=gpu_options))

DATASET_VERSION = 'v1.02-train'
DATASET_ROOT = '../data/'
ARTIFACTS_FOLDER='../data/artifacts'

level5data = LyftDataset(json_path=DATASET_ROOT + "/v1.02-train", data_path=DATASET_ROOT, verbose=True)
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

dataset=level5data
classes=[]
for i in dataset.category:
    classes.append(i.get("name"))

img_input_list=[]
img_target=[]
boxes_c=[]
class_list=[]
cam_path_list=[]
my_scene = dataset.scene[144]

# Only handle one sample at a time for now
my_sample_token = my_scene["first_sample_token"]
my_last_sample_token = my_scene["last_sample_token"]

classes=[]
for i in dataset.category:
    classes.append(i.get("name"))

with tf.device('/GPU:0'):
    while my_sample_token!=my_last_sample_token:

        clss=[]

        sample = dataset.get('sample', my_sample_token)
        sample_name = sample.get("token")
        img_data = dataset.get('sample_data', sample['data']["CAM_FRONT"])
        camera_token=img_data.get("token")

        tok=sample['data']["CAM_FRONT"]
        ego_pose = dataset.get("ego_pose", img_data["ego_pose_token"])
        cam_path, boxes_cam, camera_intrinsic = dataset.get_sample_data(camera_token)
        cam_path_list.append(cam_path)
        data = Image.open(cam_path)
        boxes_c.append(np.asarray(boxes_cam))
        for t in boxes_cam:
            clss.append(t.name)
        class_list.append(clss)

        file_name=dataset.get_sample_data_path(camera_token)
        image1 = Image.open(file_name)
        # compress image, it is too big
        # convert image to numpy array
        image = image1.resize((358,300), Image.ANTIALIAS)
        img_array = np.asarray(image)
        img_input_list.append(img_array)

        cam_front_token = dataset.get('sample_data', sample['data']["CAM_FRONT"])
        cam_front_data = cam_front_token.get("calibrated_sensor_token")
        cam_front_calib = dataset.get("calibrated_sensor", cam_front_data )
        cam_front_coords = cam_front_calib.get("translation")

        cam_front_left_token = dataset.get('sample_data', sample['data']["CAM_FRONT_LEFT"])
        cam_front_left_data = cam_front_left_token.get("calibrated_sensor_token")
        cam_front_left_calib = dataset.get("calibrated_sensor", cam_front_left_data )
        cam_front_left_coords = cam_front_left_calib.get("translation")

        cam_front_right_token = dataset.get('sample_data', sample['data']["CAM_FRONT_RIGHT"])
        cam_front_right_data = cam_front_right_token.get("calibrated_sensor_token")
        cam_front_right_calib = dataset.get("calibrated_sensor", cam_front_right_data )
        cam_front_right_coords = cam_front_right_calib.get("translation")

        ground_plane = frame_helper.get_ground_plane_coeff(cam_front_coords, cam_front_left_coords, cam_front_right_coords)
        ground_plane_list.append(ground_plane)

        token=img_data.get("calibrated_sensor_token") 
        stereo_calib_p2 = frame_helper.read_calibration(token, dataset)

        my_sample_token = dataset.get("sample", my_sample_token)["next"]
        
x=[camera_intrinsic[0], camera_intrinsic[1], camera_intrinsic[2]]
x=np.asarray(x)
gt_boxes_corners=[]
gt_boxes_corners_2=[]

for i in range(len(boxes_c)):    
    corner_img_list=[]
    corn2=[]
    for j in range(len(boxes_c[i])):
        corners = view_points(boxes_c[i][j].corners(), view=x, normalize=True)[:2, :]
        np_corners=corners.T[:4]
        corners_norm=[min(np_corners[0][0], np_corners[1][0]), min(np_corners[1][1],np_corners[2][1]),
                      max(np_corners[0][0], np_corners[1][0]), max(np_corners[2][1],np_corners[1][1])]
        corn_2=[min(corners[0]), min(corners[1]), max(corners[0], ), max(corners[1])]
        corner_img_list.append(corners_norm)
        corn2.append(corn_2)
    gt_boxes_corners.append(np.asarray(corner_img_list))
    gt_boxes_corners_2.append(corn2)

with open("annotation.txt", "a") as f:
    for i in range(len(cam_path_list)):
        #boxes_c and class_list have the same structure so its ok
        for j in range(len(class_list[i])):
            f.write(str(cam_path_list[i]) + ',' + str(gt_boxes_corners_2[i][j][0]) + ',' 
                    + str(gt_boxes_corners_2[i][j][1]) + ',' + str(gt_boxes_corners_2[i][j][2]) + ',' + str(gt_boxes_corners_2[i][j][3]) + ',' + class_list[i][j] + '\n')