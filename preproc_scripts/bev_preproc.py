from datetime import datetime
from functools import partial
import glob
#Disabled for numpy and opencv: avod has opencv and numpy versions for several methods
from multiprocessing import Pool

import os
#os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
%matplotlib inline
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

my_scene = dataset.scene[0]

        
voxel_size = (0.4,0.4,1.5)
z_offset = -2.0
#arbitrary shape, must be square though!
bev_shape = (336,336, 3)
box_scale = 0.8

my_first_sample_token = my_scene["first_sample_token"]
my_sample_token=my_first_sample_token
sample = level5data.get("sample", my_last_sample_token)

b_boxes=[]
class_list=[]
token_list=[]
while my_sample_token!=my_last_sample_token:
    c_list=[]
    gt_corners=[]
    clss=[]
    corners_voxel_list=[]
    sample = level5data.get("sample", my_sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = level5data.get("sample_data", sample_lidar_token)
    lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)
    ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
    boxes = level5data.get_boxes(sample_lidar_token)
    bev_helper.move_boxes_to_car_space(boxes, ego_pose)
    bev_helper.scale_boxes(boxes, 0.8)
    
    for box in boxes:   
        corners = box.bottom_corners()
        corners_voxel = bev_helper.car_to_voxel_coords(corners, [336,336,3], voxel_size, z_offset).transpose(1,0)
        corners_voxel = corners_voxel[:,:2] # Drop z coord
        corners_voxel_list.append(corners_voxel)
        clss.append(box.name)
    
    for l in corners_voxel_list:
        flat_list = [item for sublist in l for item in sublist]
        c_list.append(flat_list)
    
    for i in c_list:
        bottom_c=[min(i[0],i[2],i[4],i[6]), min(i[1], i[3], i[5], i[7]), max(i[0],i[2],i[4],i[6]), max(i[1], i[3], i[5], i[7])]
        gt_corners.append(bottom_c)
    b_boxes.append(gt_corners)
    class_list.append(clss)
    token_list.append(my_sample_token)
    my_sample_token = dataset.get("sample", my_sample_token)["next"]

path='G:\\lyft\\nuscenes-devkit\\data\\artifacts'
with open("annotation_bev.txt", "a") as f:
    for i in range(len(b_boxes)):
        for j in range(len(b_boxes[i])):
            base_path=os.path.join(path, "{}.png".format(token_list[i]))
            f.write(str(base_path) + ',' + str(b_boxes[i][j][0]) + ',' + str(b_boxes[i][j][1]) + ',' + str(b_boxes[i][j][2]) + ',' + str(b_boxes[i][j][3]) + ',' + class_list[i][j] + '\n')
            
