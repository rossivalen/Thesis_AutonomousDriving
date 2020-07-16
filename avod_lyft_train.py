from datetime import datetime
from functools import partial
import glob

import os
#os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm, tqdm_notebook
import argparse
import tensorflow as tf
from PIL import Image

import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

import avod
from avod.core import trainer
from avod.core import anchor_filter
from avod.core import anchor_projector
from avod.core import box_3d_encoder
from avod.core import constants
from avod.core import losses
from avod.core import model
from avod.core import summary_utils
from avod.core.label_cluster_utils import LabelClusterUtils
from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.datasets.kitti import kitti_aug
import avod.datasets.kitti.kitti_utils as kitti_utils
from avod.models.rpn_model import RpnModel
import avod.builders.config_builder_util as config_build

import anchor_helper
from frame_helper import FrameCalibrationData
import frame_helper
import preproc_helper
import bev_helper

import datetime
import os
import tensorflow as tf
import tf_agents.utils
import time

from avod.builders import optimizer_builder
from avod.core import trainer_utils
from avod.core import summary_utils

def train(model, train_config):
    """Training function for detection models.

    Args:
        model: The detection model object.
        train_config: a train_*pb2 protobuf.
            training i.e. loading RPN weights onto AVOD model.
    """

    model = model
    train_config = train_config
    # Get model configurations
    model_config = model.model_config

    # Create a variable tensor to hold the global step
    global_step_tensor = tf.Variable(
        0, trainable=False, name='global_step')

    #############################
    # Get training configurations
    #############################
    max_iterations = train_config.max_iterations
    summary_interval = train_config.summary_interval
    checkpoint_interval = train_config.checkpoint_interval
    max_checkpoints = train_config.max_checkpoints_to_keep

    checkpoint_dir = "avod/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = checkpoint_dir + '/' + model_config.checkpoint_name
    
    logdir = "avod/configs"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    

    global_summaries = set([])

    # The model should return a dictionary of predictions
    prediction_dict = model.build()

    ##############################
    # Setup loss
    ##############################
    losses_dict, total_loss = model.loss(prediction_dict)

    # Optimizer
    training_optimizer = optimizer_builder.build( train_config.optimizer, global_summaries, global_step_tensor)

    # Create the train op
    with tf.compat.v1.variable_scope('train_op'):
        #train_op = slim.learning.create_train_op(
         train_op = tf_agents.utils.eager_utils.create_train_op(
             total_loss,
             training_optimizer,
             global_step=global_step_tensor)

    # Save checkpoints regularly.
    saver = tf.compat.v1.train.Saver(max_to_keep=max_checkpoints, pad_step_number=True)

    # Add the result of the train_op to the summary
    tf.compat.v1.summary.scalar("training_loss", train_op)

    # Add maximum memory usage summary op
    # This op can only be run on device with gpu
    # so it's skipped on travis

    summaries = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))
    summary_merged = summary_utils.summaries_to_keep(summaries, global_summaries, histograms=False, input_imgs=False, input_bevs=False)

    allow_gpu_mem_growth = train_config.allow_gpu_mem_growth
   
    # GPU memory config
    gpus = tf.compat.v1.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4) 
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True, gpu_options=gpu_options))


    # Create unique folder name using datetime for summary writer
    datetime_str = str(datetime.datetime.now())
    datetime_split = datetime_str.split()
    logdir = logdir + '/train'
    train_writer = tf.compat.v1.summary.FileWriter(logdir + '/' + datetime_split[0], sess.graph)

    # Create init op
    init = tf.compat.v1.global_variables_initializer()

    # Continue from last saved checkpoint
    #if not train_config.overwrite_checkpoints:  solve later, issue in the if. runtime error idk
    if not True:
        trainer_utils.load_checkpoints(checkpoint_dir, saver)
        if len(saver.last_checkpoints) > 0:
            checkpoint_to_restore = saver.last_checkpoints[-1]
            saver.restore(sess, checkpoint_to_restore)
        else:
            # Initialize the variables
            sess.run(init)
    else:
        # Initialize the variables
        sess.run(init)

    # Read the global step if restored
    global_step = tf.compat.v1.train.global_step(sess, global_step_tensor)
    print('Starting from step {} / {}'.format(global_step, max_iterations))

    # Main Training Loop
    last_time = time.time()
    for step in range(global_step, max_iterations + 1):

        # Save checkpoint
        if step % checkpoint_interval == 0:
            global_step = tf.compat.v1.train.global_step(sess, global_step_tensor)

            saver.save(sess, save_path=checkpoint_path, global_step=global_step)

            print('Step {} / {}, Checkpoint saved to {}-{:08d}'.format(
                step, max_iterations,
                checkpoint_path, global_step))

        # Create feed_dict for inferencing
        feed_dict = model.create_feed_dict(5)

        # Write summaries and train op
        if step % summary_interval == 0:
            current_time = time.time()
            time_elapsed = current_time - last_time
            last_time = current_time

            train_op_loss, summary_out = sess.run([train_op, summary_merged], feed_dict=feed_dict)

            print('Step {}, Total Loss {:0.3f}, Time Elapsed {:0.3f} s'.format(step, train_op_loss, time_elapsed))
            train_writer.add_summary(summary_out, step)

        else:
            # Run the train op only
            sess.run(train_op, feed_dict)

    # Close the summary writers
    train_writer.close()

def main(_):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    
    DATASET_VERSION = 'v1.02-train'
    DATASET_ROOT = '../../nuscenes-devkit/data/'

    #The code will generate data, visualization and model checkpoints
    ARTIFACTS_FOLDER = "./artifacts"
    
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
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4) 
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True, gpu_options=gpu_options))
    
    level5data = LyftDataset(json_path=DATASET_ROOT + "/v1.02-train", data_path=DATASET_ROOT, verbose=True)
    os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)
    
    
    config_path = 'avod/configs/unittest_model.config'
    pipe_path = 'avod/configs/unittest_pipeline.config'
    model_config = config_build.get_model_config_from_file(config_path)
    pipeline_config=config_build.get_configs_from_pipeline_file(pipe_path, "val")

    print(pipeline_config[3].kitti_utils_config.mini_batch_config)
    rpn_model = RpnModel(model_config, pipeline_config[3],
                             train_val_test="val",
                             dataset=level5data)

    predictions = rpn_model.build()

    loss, total_loss = rpn_model.loss(predictions)

    feed_dict = rpn_model.create_feed_dict(5)
    
    train_config = pipeline_config[1]
    train(rpn_model, train_config)
if __name__ == '__main__':
    tf.app.run()