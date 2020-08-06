import tensorflow as tf
import numpy as np

from PIL import Image
from avod.core import anchor_filter
from avod.core import anchor_projector
from avod.core import box_3d_encoder
from avod.core import constants
from avod.core import losses
from avod.core import model
from avod.core import summary_utils
from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.datasets.kitti import kitti_aug
import avod.datasets.kitti.kitti_utils as kitti_utils
from avod.core.label_cluster_utils import LabelClusterUtils
from utils import anchor_helper
from utils import frame_helper
from utils import preproc_helper
from utils import bev_helper

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


class RpnModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    PL_BEV_INPUT = 'bev_input_pl'
    PL_IMG_INPUT = 'img_input_pl'
    PL_ANCHORS = 'anchors_pl'

    PL_BEV_ANCHORS = 'bev_anchors_pl'
    PL_BEV_ANCHORS_NORM = 'bev_anchors_norm_pl'
    PL_IMG_ANCHORS = 'img_anchors_pl'
    PL_IMG_ANCHORS_NORM = 'img_anchors_norm_pl'
    PL_LABEL_ANCHORS = 'label_anchors_pl'
    PL_LABEL_BOXES_3D = 'label_boxes_3d_pl'
    PL_LABEL_CLASSES = 'label_classes_pl'

    PL_ANCHOR_IOUS = 'anchor_ious_pl'
    PL_ANCHOR_OFFSETS = 'anchor_offsets_pl'
    PL_ANCHOR_CLASSES = 'anchor_classes_pl'

    # Sample info, including keys for projection to image space
    # (e.g. camera matrix, image index, etc.)
    PL_CALIB_P2 = 'frame_calib_p2'
    PL_IMG_IDX = 'current_img_idx'
    PL_GROUND_PLANE = 'ground_plane'

    ##############################
    # Keys for Predictions
    ##############################
    PRED_ANCHORS = 'rpn_anchors'

    PRED_MB_OBJECTNESS_GT = 'rpn_mb_objectness_gt'
    PRED_MB_OFFSETS_GT = 'rpn_mb_offsets_gt'

    PRED_MB_MASK = 'rpn_mb_mask'
    PRED_MB_OBJECTNESS = 'rpn_mb_objectness'
    PRED_MB_OFFSETS = 'rpn_mb_offsets'

    PRED_TOP_INDICES = 'rpn_top_indices'
    PRED_TOP_ANCHORS = 'rpn_top_anchors'
    PRED_TOP_OBJECTNESS_SOFTMAX = 'rpn_top_objectness_softmax'

    ##############################
    # Keys for Loss
    ##############################
    LOSS_RPN_OBJECTNESS = 'rpn_objectness_loss'
    LOSS_RPN_REGRESSION = 'rpn_regression_loss'

    def __init__(self, model_config, pipeline_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(RpnModel, self).__init__(model_config)
        self.dataset=dataset
        self.pipeline_config = pipeline_config

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test

        self._is_training = (self._train_val_test == 'train')

        # Input config
        input_config = self._config.input_config
        self._bev_pixel_size = np.asarray([input_config.bev_dims_h,
                                           input_config.bev_dims_w])
        self._bev_depth = input_config.bev_depth

        self._img_pixel_size = np.asarray([input_config.img_dims_h,
                                           input_config.img_dims_w])
        self._img_depth = input_config.img_depth

        # Rpn config
        rpn_config = self._config.rpn_config
        self.proposal_roi_crop_size = 3*2  #3*2
        self._fusion_method = rpn_config.rpn_fusion_method

        if self._train_val_test in ["train", "val"]:
            self._nms_size = rpn_config.rpn_train_nms_size
        else:
            self._nms_size = rpn_config.rpn_test_nms_size

        self._nms_iou_thresh = rpn_config.rpn_nms_iou_thresh

        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()

        # Information about the current sample
        self.sample_info = dict()

        # Dataset
        classes=[]
        for i in self.dataset.category:
            classes.append(i.get("name"))
        self.classes = classes
        self.dataset = dataset
        self.dataset.train_val_test = self._train_val_test
        area_extents = self.pipeline_config.kitti_utils_config.area_extents
        self._area_extents = np.reshape(area_extents, (3, 2))
        self._bev_extents = self._area_extents[[0, 2]]
        
        label_cluster_utils = LabelClusterUtils(self.dataset)
        self._cluster_sizes, self._all_std = label_cluster_utils.get_clusters(5, self.dataset)
        
        anchor_strides = self.pipeline_config.kitti_utils_config.anchor_strides
        self._anchor_strides= np.reshape(anchor_strides, (-1, 2))
        self._anchor_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()

        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._train_on_all_samples = self._config.train_on_all_samples
        self._eval_all_samples = self._config.eval_all_samples

        if self._train_val_test in ["val", "test"]:
            # Disable path-drop, this should already be disabled inside the
            # evaluator, but just in case.
            self._path_drop_probabilities[0] = 1.0
            self._path_drop_probabilities[1] = 1.0

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.compat.v1.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        # Combine config data
        bev_dims = np.append(self._bev_pixel_size, self._bev_depth)

        with tf.compat.v1.variable_scope('bev_input'):
            # Placeholder for BEV image input, to be filled in with feed_dict
            bev_input_placeholder = self._add_placeholder(tf.float32, bev_dims,
                                                          self.PL_BEV_INPUT)

            self._bev_input_batches = tf.expand_dims(
                bev_input_placeholder, axis=0)

            self._bev_preprocessed = tf.image.resize(self._bev_input_batches, self._bev_pixel_size)

            # Summary Images
            bev_summary_images = tf.split(bev_input_placeholder, self._bev_depth, axis=2)
            tf.summary.image("bev_maps", bev_summary_images, max_outputs=self._bev_depth)

        with tf.compat.v1.variable_scope('img_input'):
            # Take variable size input images
            img_input_placeholder = self._add_placeholder(tf.float32, [None, None, self._img_depth],self.PL_IMG_INPUT)

            self._img_input_batches = tf.expand_dims(img_input_placeholder, axis=0)

            self._img_preprocessed = tf.image.resize(self._img_input_batches, self._img_pixel_size)

            # Summary Image
            tf.summary.image("rgb_image", self._img_preprocessed, max_outputs=2)

        with tf.compat.v1.variable_scope('pl_labels'):
            #self._add_placeholder(tf.float32, [None, 6], self.PL_LABEL_ANCHORS)
            self._add_placeholder(tf.float32, [None, 7], self.PL_LABEL_BOXES_3D)
            #self._add_placeholder(tf.float32, [None], self.PL_LABEL_CLASSES)

        # Placeholders for anchors
        with tf.compat.v1.variable_scope('pl_anchors'):
            self._add_placeholder(tf.float32, [None, 6], self.PL_ANCHORS)
            self._add_placeholder(tf.float32, [None], self.PL_ANCHOR_IOUS)
            self._add_placeholder(tf.float32, [None, 6], self.PL_ANCHOR_OFFSETS)
            self._add_placeholder(tf.float32, [None], self.PL_ANCHOR_CLASSES)

            with tf.compat.v1.variable_scope('bev_anchor_projections'):
                self._add_placeholder(tf.float32, [None, 4], self.PL_BEV_ANCHORS)
                self._bev_anchors_norm_pl = self._add_placeholder( tf.float32, [None, 4], self.PL_BEV_ANCHORS_NORM)

            with tf.compat.v1.variable_scope('img_anchor_projections'):
                self._add_placeholder(tf.float32, [None, 4], self.PL_IMG_ANCHORS)
                self._img_anchors_norm_pl = self._add_placeholder( tf.float32, [None, 4], self.PL_IMG_ANCHORS_NORM)

            with tf.compat.v1.variable_scope('sample_info'):
                # the calib matrix shape is (3 x 4)
                self._add_placeholder( tf.float32, [3, 4], self.PL_CALIB_P2)
                self._add_placeholder(tf.int32, shape=[1], name=self.PL_IMG_IDX)
                self._add_placeholder(tf.float32, [4], self.PL_GROUND_PLANE)

    def _set_up_feature_extractors(self):
        """Sets up feature extractors and stores feature maps and
        bottlenecks as member variables.
        """
        weight_decay=0.0005
        #shape due to shape provided by dataset. BEV could not be adapted: too sparse.
        
        inputs_img = tf.keras.layers.Input(batch_shape=(None,1024,1224,3))
        net = tf.keras.layers.Conv2D(filters = 32, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), 
                                     activation=tf.nn.relu, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv1")(inputs_img)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch1")(net)
        net = tf.keras.layers.Conv2D(filters = 32, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
                                     use_bias=False, padding='same', name="conv2")(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch2")(net)   

        net = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid", name="pool1")(net)

        net = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv3")(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch3")(net)
        net = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv4")(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch4")(net)

        net = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid", name="pool2")(net)

        net = tf.keras.layers.Conv2D(filters = 128,kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv5")(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch5")(net)
        net = tf.keras.layers.Conv2D(filters = 128, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
                                     use_bias=False, padding='same', name="conv6")(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch6")(net)
        net = tf.keras.layers.Conv2D(filters = 128, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv7")(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch7")(net)

        net = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid", name="pool3")(net)

        net = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv8") (net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch8")(net)
        net = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv9")(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch9")(net)
        net = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv10")(net)
        net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch10")(net)
        
        img_vgg = tf.keras.models.Model(inputs = inputs_img, outputs = net, name="img_vgg")
        
        self.img_bottleneck = tf.keras.layers.Conv2D(filters = 32, kernel_size = [1,1], strides =(1,1), padding='same', name="bottleneck")(net)
        self.img_bottleneck= tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                                            gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                                            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(self.img_bottleneck)
        
         #shape due to shape provided by dataset. BEV could not be adapted: too sparse.
        
        inputs_bev = tf.keras.layers.Input(batch_shape=(None,336,336,3))
        out = tf.keras.layers.Conv2D(filters = 32, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), 
                                     activation=tf.nn.relu, kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv1")(inputs_bev)
        out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch1")(out)
        out = tf.keras.layers.Conv2D(filters = 32, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
                                     use_bias=False, padding='same', name="conv2")(out)
        out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch2")(out)   

        out = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid", name="pool1")(out)

        out = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv3")(out)
        out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch3")(out)
        out = tf.keras.layers.Conv2D(filters = 64, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv4")(out)
        out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch4")(out)

        out = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid", name="pool2")(out)

        out = tf.keras.layers.Conv2D(filters = 128,kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv5")(out)
        out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch5")(out)
        out = tf.keras.layers.Conv2D(filters = 128, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
                                     use_bias=False, padding='same', name="conv6")(out)
        out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch6")(out)
        out = tf.keras.layers.Conv2D(filters = 128, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv7")(out)
        out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch7")(out)

        out = tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=None, padding="valid", name="pool3")(out)

        out = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv8") (out)
        out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch8")(out)
        out = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv9")(out)
        out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch9")(out)
        out = tf.keras.layers.Conv2D(filters = 256, kernel_size = [3,3], strides =(1,1), bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu,
                                     kernel_initializer='ones', kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                     use_bias=False, padding='same', name="conv10")(out)
        out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                        beta_constraint=None, gamma_constraint=None, name="batch10")(out)
        
        bev_vgg = tf.keras.models.Model(inputs = inputs_bev, outputs = out, name="bev_vgg")
        
        with tf.compat.v1.variable_scope("bev_bottleneck"):
            self.bev_bottleneck = tf.keras.layers.Conv2D(filters = 32, kernel_size = [1,1], strides =(1,1), padding='same', name="bottleneck")(out)
        with tf.compat.v1.variable_scope("img_bottleneck"):
            self.bev_bottleneck= tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                                            gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                                            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(self.bev_bottleneck)

        # # Visualize the end point feature maps being used
        bev_vgg.summary()
        bev_end_point=bev_vgg.get_config()
        img_vgg.summary()
        img_end_point=img_vgg.get_config()

    def build(self):

        # Setup input placeholders
        self._set_up_input_pls()

        # Setup feature extractors
        self._set_up_feature_extractors()

        bev_proposal_input = self.bev_bottleneck
        img_proposal_input = self.img_bottleneck

        fusion_mean_div_factor = 2.0

        # If both img and bev probabilites are set to 1.0, don't do
        # path drop.
        if not (self._path_drop_probabilities[0] ==
                self._path_drop_probabilities[1] == 1.0):
            with tf.compat.v1.variable_scope('rpn_path_drop'):

                random_values = tf.random_uniform(shape=[3],
                                                  minval=0.0,
                                                  maxval=1.0)

                img_mask, bev_mask = self.create_path_drop_masks(
                    self._path_drop_probabilities[0],
                    self._path_drop_probabilities[1],
                    random_values)

                img_proposal_input = tf.multiply(img_proposal_input,
                                                 img_mask)

                bev_proposal_input = tf.multiply(bev_proposal_input,
                                                 bev_mask)

                self.img_path_drop_mask = img_mask
                self.bev_path_drop_mask = bev_mask

                # Overwrite the division factor
                fusion_mean_div_factor = img_mask + bev_mask

        with tf.compat.v1.variable_scope('proposal_roi_pooling'):

            with tf.compat.v1.variable_scope('box_indices'):
                def get_box_indices(boxes):
                    proposals_shape = boxes.get_shape().as_list()
                    if any(dim is None for dim in proposals_shape):
                        proposals_shape = tf.shape(boxes)
                    ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                    multiplier = tf.expand_dims(
                        tf.range(start=0, limit=proposals_shape[0]), 1)
                    return tf.reshape(ones_mat * multiplier, [-1])

                bev_boxes_norm_batches = tf.expand_dims(
                    self._bev_anchors_norm_pl, axis=0)

                # These should be all 0's since there is only 1 image
                tf_box_indices = get_box_indices(bev_boxes_norm_batches)
            
            proposal_roi_size_tf = [3,3]
            # Do ROI Pooling on BEV
            bev_proposal_rois = tf.image.crop_and_resize(
                bev_proposal_input,
                self._bev_anchors_norm_pl,
                tf_box_indices,
                proposal_roi_size_tf)
            # Do ROI Pooling on image
            img_proposal_rois = tf.image.crop_and_resize(
                img_proposal_input,
                self._bev_anchors_norm_pl,
                tf_box_indices,
                proposal_roi_size_tf)

        with tf.compat.v1.variable_scope('proposal_roi_fusion'):
            rpn_fusion_out = None
            if self._fusion_method == 'mean':
                tf_features_sum = tf.add(bev_proposal_rois, img_proposal_rois)
                #rpn_fusion_out = tf.divide(tf_features_sum, fusion_mean_div_factor)
                rpn_fusion_out = tf.divide(tf_features_sum, 2)
            elif self._fusion_method == 'concat':
                rpn_fusion_out = tf.concat(
                    [bev_proposal_rois, img_proposal_rois], axis=3)
            else:
                raise ValueError('Invalid fusion method', self._fusion_method)

        with tf.compat.v1.variable_scope('anchor_predictor', 'ap', [rpn_fusion_out]):
            #None because unknown
            tensor_in = tf.keras.Input(shape=None, tensor=rpn_fusion_out)
            print("here", tf_features_sum)
            # Rpn layers config
            weight_decay = 0.005

            # Use conv2d instead of fully_connected layers.
            cls_fc6 = tf.keras.layers.Conv2D(filters=32, kernel_size = [3,3], kernel_initializer='ones', 
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='VALID', name="conv1")(tensor_in)

            cls_fc6_drop = tf.keras.layers.Dropout(rate = 0.5, name="drop1")(cls_fc6)

            cls_fc7 = tf.keras.layers.Conv2D(filters=32, kernel_size = [1,1], kernel_initializer='ones',
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='VALID', name="conv2")(cls_fc6_drop)

            cls_fc7_drop = tf.keras.layers.Dropout(rate = 0.5, name="drop2")(cls_fc7)

            cls_fc8 = tf.keras.layers.Conv2D(filters=2, kernel_size = [1,1], kernel_initializer='ones',
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='VALID', name="conv3")(cls_fc7_drop)

            objectness = tf.squeeze(cls_fc8, axis=[1,2], name='conv3/squeezed')

            # Use conv2d instead of fully_connected layers.
            reg_fc6 = tf.keras.layers.Conv2D(filters=32, kernel_size = [3,3], kernel_initializer="ones", 
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='VALID', name="conv4")(tensor_in)

            reg_fc6_drop = tf.keras.layers.Dropout(rate = 0.5, name="drop3")(reg_fc6)

            reg_fc7 = tf.keras.layers.Conv2D(filters = 16, kernel_size = [1, 1], kernel_initializer="ones",
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding="same", name="conv5")(reg_fc6_drop)

            reg_fc7_drop = tf.keras.layers.Dropout(rate = 0.5, name="drop4")(reg_fc7)

            reg_fc8 = tf.keras.layers.Conv2D(filters = 6,  kernel_size = [1, 1],  kernel_initializer="ones",
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding="same", name="conv6")(reg_fc7_drop)

            offsets = tf.squeeze(reg_fc8, axis=[1,2], name='conv6/squeezed')
            
            model = tf.keras.models.Model(inputs = rpn_fusion_out, outputs = offsets, name="rpn_fusion_prediction_anchors")
            model1 = tf.keras.models.Model(inputs = rpn_fusion_out, outputs = objectness, name="objectness predictions")
            model.summary()
            model1.summary()
        # Return the proposals
        with tf.compat.v1.variable_scope('proposals'):
            anchors = self.placeholders[self.PL_ANCHORS]

            # Decode anchor regression offsets
            with tf.compat.v1.variable_scope('decoding'):
                regressed_anchors = anchor_helper.offset_to_anchor( anchors, offsets)

            with tf.compat.v1.variable_scope('bev_projection'):
                _, bev_proposal_boxes_norm = anchor_projector.project_to_bev(regressed_anchors, self._bev_extents)

            with tf.compat.v1.variable_scope('softmax'):
                objectness_softmax = tf.nn.softmax(objectness)

            with tf.compat.v1.variable_scope('nms'):
                objectness_scores = objectness_softmax[:, 1]

                # Do NMS on regressed anchors
                top_indices = tf.image.non_max_suppression(
                    bev_proposal_boxes_norm, objectness_scores,
                    max_output_size=self._nms_size,
                    iou_threshold=self._nms_iou_thresh)

                top_anchors = tf.gather(regressed_anchors, top_indices)
                top_objectness_softmax = tf.gather(objectness_scores,
                                                   top_indices)
                # top_offsets = tf.gather(offsets, top_indices)
                # top_objectness = tf.gather(objectness, top_indices)

        # Get mini batch
        all_ious_gt = self.placeholders[self.PL_ANCHOR_IOUS]
        all_offsets_gt = self.placeholders[self.PL_ANCHOR_OFFSETS]
        all_classes_gt = self.placeholders[self.PL_ANCHOR_CLASSES]

        with tf.compat.v1.variable_scope('mini_batch'):
            mini_batch_mask, _ = anchor_helper.sample_mini_batch(all_ious_gt, 64,[0, 0.3], [0.5,1])

        # ROI summary images
        rpn_mini_batch_size =64
        with tf.compat.v1.variable_scope('bev_rpn_rois'):
            mb_bev_anchors_norm = tf.boolean_mask(self._bev_anchors_norm_pl,
                                                  mini_batch_mask)
            mb_bev_box_indices = tf.zeros_like(
                tf.boolean_mask(all_classes_gt, mini_batch_mask),
                dtype=tf.int32)

            # Show the ROIs of the BEV input density map
            # for the mini batch anchors
            bev_input_rois = tf.image.crop_and_resize(self._bev_preprocessed,
                                                      mb_bev_anchors_norm, mb_bev_box_indices, (32, 32))

            bev_input_roi_summary_images = tf.split(bev_input_rois, self._bev_depth, axis=3)
            tf.summary.image('bev_rpn_rois', bev_input_roi_summary_images[-1], max_outputs=rpn_mini_batch_size)

        with tf.compat.v1.variable_scope('img_rpn_rois'):
            # ROIs on image input
            mb_img_anchors_norm = tf.boolean_mask(self._img_anchors_norm_pl, mini_batch_mask)
            mb_img_box_indices = tf.zeros_like( tf.boolean_mask(all_classes_gt, mini_batch_mask), dtype=tf.int32)

            # Do test ROI pooling on mini batch
            img_input_rois = tf.image.crop_and_resize( self._img_preprocessed,
                                                      mb_img_anchors_norm, mb_img_box_indices, (32, 32))

            tf.summary.image('img_rpn_rois', img_input_rois, max_outputs=rpn_mini_batch_size)

        # Ground Truth Tensors
        with tf.compat.v1.variable_scope('one_hot_classes'):

            # Anchor classification ground truth
            # Object / Not Object
            min_pos_iou = 0.5

            objectness_classes_gt = tf.cast(tf.greater_equal(all_ious_gt, min_pos_iou), dtype=tf.int32)
            objectness_gt = tf.one_hot(objectness_classes_gt, depth=2, on_value=1.0 - self._config.label_smoothing_epsilon,
                                       off_value=self._config.label_smoothing_epsilon)

        # Mask predictions for mini batch
        with tf.compat.v1.variable_scope('prediction_mini_batch'):
            objectness_masked = tf.boolean_mask(objectness, mini_batch_mask)
            offsets_masked = tf.boolean_mask(offsets, mini_batch_mask)

        with tf.compat.v1.variable_scope('ground_truth_mini_batch'):
            objectness_gt_masked = tf.boolean_mask(objectness_gt, mini_batch_mask)
            offsets_gt_masked = tf.boolean_mask(all_offsets_gt, mini_batch_mask)

        # Specify the tensors to evaluate
        predictions = dict()

        # Temporary predictions for debugging
#         predictions['anchor_ious'] = anchor_ious
#         predictions['anchor_offsets'] = all_offsets_gt

        if self._train_val_test in ['train', 'val']:
            # All anchors
            predictions[self.PRED_ANCHORS] = anchors

            # Mini-batch masks
            predictions[self.PRED_MB_MASK] = mini_batch_mask
            # Mini-batch predictions
            predictions[self.PRED_MB_OBJECTNESS] = objectness_masked
            predictions[self.PRED_MB_OFFSETS] = offsets_masked

            # Mini batch ground truth
            predictions[self.PRED_MB_OFFSETS_GT] = offsets_gt_masked
            predictions[self.PRED_MB_OBJECTNESS_GT] = objectness_gt_masked

            # Proposals after nms
            predictions[self.PRED_TOP_INDICES] = top_indices
            predictions[self.PRED_TOP_ANCHORS] = top_anchors
            predictions[
                self.PRED_TOP_OBJECTNESS_SOFTMAX] = top_objectness_softmax

        else:
            # self._train_val_test == 'test'
            predictions[self.PRED_TOP_ANCHORS] = top_anchors
            predictions[
                self.PRED_TOP_OBJECTNESS_SOFTMAX] = top_objectness_softmax

        return predictions

    def create_feed_dict(self, scene_index=None):
        """ Fills in the placeholders with the actual input values.
            Currently, only a batch size of 1 is supported

        Args:
            sample_index: optional, only used when train_val_test == 'test',
                a particular sample index in the dataset
                sample list to build the feed_dict for

        Returns:
            a feed_dict dictionary that can be used in a tensorflow session
        """
#TODO fix to have multiple batches
#         if self._train_val_test in ["train", "val"]:

#             # sample_index should be None
#             if sample_index is not None:
#                 raise ValueError('sample_index should be None. Do not load '
#                                  'particular samples during train or val')

#             # During training/validation, we need a valid sample
#             # with anchor info for loss calculation
#             sample = None
#             anchors_info = []

#             valid_sample = False
#             while not valid_sample:
#                 if self._train_val_test == "train":
#                     # Get the a random sample from the remaining epoch
#                     samples = self.dataset.next_batch(batch_size=1)

#                 else:  # self._train_val_test == "val"
#                     # Load samples in order for validation
#                     samples = self.dataset.next_batch(batch_size=1, shuffle=False)

#                 # Only handle one sample at a time for now
#                 sample = samples[0]
#                 anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

#                 # When training, if the mini batch is empty, go to the next
#                 # sample. Otherwise carry on with found the valid sample.
#                 # For validation, even if 'anchors_info' is empty, keep the
#                 # sample (this will help penalize false positives.)
#                 # We will substitue the necessary info with zeros later on.
#                 # Note: Training/validating all samples can be switched off.
#                 train_cond = (self._train_val_test == "train" and self._train_on_all_samples)
#                 eval_cond = (self._train_val_test == "val" and self._eval_all_samples)
#                 if anchors_info or train_cond or eval_cond:
#                     valid_sample = True
#         else:
        # For testing, any sample should work
        if scene_index is not None:
            my_scene = self.dataset.scene[scene_index]
        else:
            raise TypeError('for testing you need to put a number! will change it later on once it works fully :) ')
    
        
        # Only handle one sample at a time for now
        my_sample_token = my_scene["first_sample_token"]
        sample = self.dataset.get('sample', my_sample_token)
        sample_name = sample.get("token")
        
        last_sample_token = my_scene["last_sample_token"]

        # We only need orientation from box_3
        anchors_info, obj_classes, label_classes, label_anchors, label_boxes_3d = preproc_helper.load_sample_info(sample_name, self.classes, self.dataset)
        
        # Network input data
        img_input = self.dataset.get('sample_data', sample['data']["CAM_FRONT"])
        img_data=img_input
        camera_token=img_input.get("token")
        file_name=self.dataset.get_sample_data_path(camera_token)
        image = Image.open(file_name)
        # convert image to numpy array
        img_input = np.asarray(image)
        bev_input = self.dataset.get('sample_data', sample['data']["LIDAR_TOP"])
        bev_data = bev_input
        bev_token= bev_input.get("token")
        lidar_data = self.dataset.get("sample_data", bev_token)
        lidar_filepath = self.dataset.get_sample_data_path(bev_token)
        ego_pose = self.dataset.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor_lidar = self.dataset.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        global_from_car = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
        car_from_sensor_lidar = transform_matrix(calibrated_sensor_lidar['translation'], Quaternion(calibrated_sensor_lidar['rotation']),
                                                  inverse=False)
        lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
        lidar_pointcloud.transform(car_from_sensor_lidar)
        map_mask = level5data.map[0]["mask"]
        voxel_size = (0.4,0.4,1.5)
        z_offset = -2.0
        #arbitrary shape, must be square though!
        bev_shape = (336,336, 3)
        bev = bev_helper.create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
        ego_centric_map = bev_helper.get_semantic_map_around_ego(map_mask, ego_pose, voxel_size=0.4, output_shape=(336,336)) 
        bev_input = bev_helper.normalize_voxel_intensities(bev)

        # Image shape (h, w)
        image_shape = [img_data.get("height"), img_data.get("width")]
        
        #ground plane shape (a,b,c,d) in kitti:
        #no info on ground plane in nuscenes data, just global coordinate system
        #which is given as x, y, z. Computed from the cameras position:
        #suppose as in kitti that ground plane is as the same level with the cameras
                       
        cam_front_token = self.dataset.get('sample_data', sample['data']["CAM_FRONT"])
        cam_front_data = cam_front_token.get("calibrated_sensor_token")
        cam_front_calib = self.dataset.get("calibrated_sensor", cam_front_data )
        cam_front_coords = cam_front_calib.get("translation")

        cam_front_left_token = self.dataset.get('sample_data', sample['data']["CAM_FRONT_LEFT"])
        cam_front_left_data = cam_front_left_token.get("calibrated_sensor_token")
        cam_front_left_calib = self.dataset.get("calibrated_sensor", cam_front_left_data )
        cam_front_left_coords = cam_front_left_calib.get("translation")

        cam_front_right_token = self.dataset.get('sample_data', sample['data']["CAM_FRONT_RIGHT"])
        cam_front_right_data = cam_front_right_token.get("calibrated_sensor_token")
        cam_front_right_calib = self.dataset.get("calibrated_sensor", cam_front_right_data )
        cam_front_right_coords = cam_front_right_calib.get("translation")
        
        ground_plane = frame_helper.get_ground_plane_coeff(cam_front_coords, cam_front_left_coords, cam_front_right_coords)
        
        #only for cameras, of course lidars do not have instrinsic matrices
        token=img_data.get("calibrated_sensor_token") 
        stereo_calib_p2 = frame_helper.read_calibration(token, self.dataset)

        # Fill the placeholders for anchor information
        self._fill_anchor_pl_inputs(anchors_info=anchors_info,sample_token=bev_token, ground_plane=ground_plane,
                                    image_shape=image_shape, stereo_calib_p2=stereo_calib_p2,
                                    sample_name=sample_name)

        # Fill in the rest
        self._placeholder_inputs[self.PL_BEV_INPUT] = bev_input
        self._placeholder_inputs[self.PL_IMG_INPUT] = img_input

        self._placeholder_inputs[self.PL_LABEL_ANCHORS] = label_anchors
        self._placeholder_inputs[self.PL_LABEL_BOXES_3D] = label_boxes_3d
        self._placeholder_inputs[self.PL_LABEL_CLASSES] = label_classes

        # Sample Info
        # img_idx is a list to match the placeholder shape
        self._placeholder_inputs[self.PL_IMG_IDX] = [str(sample_name)]
        self._placeholder_inputs[self.PL_CALIB_P2] = stereo_calib_p2
        self._placeholder_inputs[self.PL_GROUND_PLANE] = ground_plane

        # Temporary sample info for debugging
        self.sample_info.clear()
        self.sample_info['sample_name'] = sample
        self.sample_info['rpn_mini_batch'] = anchors_info

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]

        return feed_dict

    def _fill_anchor_pl_inputs(self,
                               sample_token,
                               anchors_info,
                               ground_plane,
                               image_shape,
                               stereo_calib_p2,
                               sample_name):
        """
        Fills anchor placeholder inputs with corresponding data

        Args:
            anchors_info: anchor info from mini_batch_utils
            ground_plane: ground plane coefficients
            image_shape: image shape (h, w), used for projecting anchors
            sample_name: name of the sample, e.g. "000001"
        """

        # Lists for merging anchors info
        all_anchor_boxes_3d = []
        anchors_ious = []
        anchor_offsets = []
        anchor_classes = []
        
        # Create anchors for each class
        if len(self.classes) > 1:
            for class_idx in range(len(self.classes)):
                cluster_sizes = []
                for i in self._cluster_sizes[class_idx]:
                    if i!=[]:
                        cluster_sizes.append(i)
                if len(cluster_sizes)!=0:
                # Generate anchors for all classes
                    grid_anchor_boxes_3d = self._anchor_generator.generate(
                        area_3d=self._area_extents,
                        anchor_3d_sizes=cluster_sizes,
                        anchor_stride=self._anchor_strides[0],
                        ground_plane=ground_plane)
                else:
                    #no labels per class, no anchor per class
                    grid_anchor_boxes_3d=[]
                all_anchor_boxes_3d.append(grid_anchor_boxes_3d)
#             length=[]
#             for i in grid_anchor_boxes_3d:
#                 length.append(len(i))
#             if not all(i==length[0] for i in length):
#                 max_length=np.amax(length)
#                 for all 
            
            all_anchor_boxes_3d = np.concatenate(all_anchor_boxes_3d, axis=None)
        else:
            # Don't loop for a single class
            class_idx = 0
            cluster_sizes[class_idx] = [x for x in self._cluster_sizes[class_idx] if x != []]
            if self._cluster_sizes[class_idx]!=[]:
                grid_anchor_boxes_3d = self._anchor_generator.generate(
                    area_3d=self._area_extents,
                    anchor_3d_sizes=cluster_sizes[class_idx],
                    anchor_stride=self._anchor_strides[0],
                    ground_plane=ground_plane)
                all_anchor_boxes_3d = grid_anchor_boxes_3d

        # Filter empty anchors
        # Skip if anchors_info is []
        sample_has_labels = True

        # Convert lists to ndarrays
        #already filtered them before
        anchor_boxes_3d_to_use = all_anchor_boxes_3d
        anchors_ious = np.asarray(anchors_ious)
        anchor_offsets = np.asarray(anchor_offsets)
        anchor_classes = np.asarray(anchor_classes)

        # Flip anchors and centroid x offsets for augmented samples
#             if kitti_aug.AUG_FLIPPING in sample_augs:
#                 anchor_boxes_3d_to_use = kitti_aug.flip_boxes_3d(anchor_boxes_3d_to_use, flip_ry=False)
#                 if anchors_info:
#                     anchor_offsets[:, 0] = -anchor_offsets[:, 0]

        # Convert to anchors
        anchors_to_use = box_3d_encoder.box_3d_to_anchor( anchor_boxes_3d_to_use)
        num_anchors = len(anchors_to_use)

        # Project anchors into bev
        print(anchors_to_use.shape, self._bev_extents)
        bev_anchors, bev_anchors_norm = anchor_projector.project_to_bev( anchors_to_use, self._bev_extents)

        # Project box_3d anchors into image space
        img_anchors, img_anchors_norm = anchor_projector.project_to_image_space(anchors_to_use, stereo_calib_p2, image_shape)

        # Reorder into [y1, x1, y2, x2] for tf.crop_and_resize op
        self._bev_anchors_norm = bev_anchors_norm[:, [1, 0, 3, 2]]
        self._img_anchors_norm = img_anchors_norm[:, [1, 0, 3, 2]]

        # Fill in placeholder inputs
        self._placeholder_inputs[self.PL_ANCHORS] = anchors_to_use

        # If we are in train/validation mode, and the anchor infos
        # are not empty, store them. Checking for just anchors_ious
        # to be non-empty should be enough.
        if self._train_val_test in ['train', 'val'] and \
                len(anchors_ious) > 0:
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = anchors_ious
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = anchor_offsets
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = anchor_classes

        # During test, or val when there is no anchor info
        elif self._train_val_test in ['test'] or \
                len(anchors_ious) == 0:
            # During testing, or validation with no gt, fill these in with 0s
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = \
                np.zeros(num_anchors)
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = \
                np.zeros([num_anchors, 6])
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = \
                np.zeros(num_anchors)
        else:
            raise ValueError('Got run mode {}, and non-empty anchor info'.
                             format(self._train_val_test))

        self._placeholder_inputs[self.PL_BEV_ANCHORS] = bev_anchors
        self._placeholder_inputs[self.PL_BEV_ANCHORS_NORM] = self._bev_anchors_norm
        self._placeholder_inputs[self.PL_IMG_ANCHORS] = img_anchors
        self._placeholder_inputs[self.PL_IMG_ANCHORS_NORM] = self._img_anchors_norm

    def loss(self, prediction_dict):

        # these should include mini-batch values only
        objectness_gt = prediction_dict[self.PRED_MB_OBJECTNESS_GT]
        offsets_gt = prediction_dict[self.PRED_MB_OFFSETS_GT]

        # Predictions
        with tf.compat.v1.variable_scope('rpn_prediction_mini_batch'):
            objectness = prediction_dict[self.PRED_MB_OBJECTNESS]
            offsets = prediction_dict[self.PRED_MB_OFFSETS]

        with tf.compat.v1.variable_scope('rpn_losses'):
            with tf.compat.v1.variable_scope('objectness'):
                cls_loss = losses.WeightedSoftmaxLoss()
                cls_loss_weight = self._config.loss_config.cls_loss_weight
                objectness_loss = cls_loss(objectness, objectness_gt, weight=cls_loss_weight)

                with tf.compat.v1.variable_scope('obj_norm'):
                    # normalize by the number of anchor mini-batches
                    objectness_loss = objectness_loss / tf.cast( tf.shape(objectness_gt)[0], dtype=tf.float32)
                    tf.summary.scalar('objectness', objectness_loss)

            with tf.compat.v1.variable_scope('regression'):
                reg_loss = losses.WeightedSmoothL1Loss()
                reg_loss_weight = self._config.loss_config.reg_loss_weight
                anchorwise_localization_loss = reg_loss(offsets, offsets_gt, weight=reg_loss_weight)
                masked_localization_loss = anchorwise_localization_loss * objectness_gt[:, 1]
                localization_loss = tf.reduce_sum(masked_localization_loss)

                with tf.compat.v1.variable_scope('reg_norm'):
                    # normalize by the number of positive objects
                    num_positives = tf.reduce_sum(objectness_gt[:, 1])
                    # Assert the condition `num_positives > 0`
                    with tf.control_dependencies([tf.debugging.assert_positive(num_positives)]):
                        localization_loss = localization_loss / num_positives
                        tf.summary.scalar('regression', localization_loss)

            with tf.compat.v1.variable_scope('total_loss'):
                total_loss = objectness_loss + localization_loss

        loss_dict = {
            self.LOSS_RPN_OBJECTNESS: objectness_loss,
            self.LOSS_RPN_REGRESSION: localization_loss,
        }

        return loss_dict, total_loss

    def create_path_drop_masks(self,
                               p_img,
                               p_bev,
                               random_values):
        """Determines global path drop decision based on given probabilities.

        Args:
            p_img: A tensor of float32, probability of keeping image branch
            p_bev: A tensor of float32, probability of keeping bev branch
            random_values: A tensor of float32 of shape [3], the results
                of coin flips, values should range from 0.0 - 1.0.

        Returns:
            final_img_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
            final_bev_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
        """

        def keep_branch(): return tf.constant(1.0)

        def kill_branch(): return tf.constant(0.0)

        # The logic works as follows:
        # We have flipped 3 coins, first determines the chance of keeping
        # the image branch, second determines keeping bev branch, the third
        # makes the final decision in the case where both branches were killed
        # off, otherwise the initial img and bev chances are kept.

        img_chances = tf.case([(tf.less(random_values[0], p_img), keep_branch)], default=kill_branch)

        bev_chances = tf.case([(tf.less(random_values[1], p_bev), keep_branch)], default=kill_branch)

        # Decision to determine whether both branches were killed off
        third_flip = tf.logical_or(tf.cast(img_chances, dtype=tf.bool), tf.cast(bev_chances, dtype=tf.bool))
        third_flip = tf.cast(third_flip, dtype=tf.float32)

        # Make a second choice, for the third case
        # Here we use a 50/50 chance to keep either image or bev
        # If its greater than 0.5, keep the image
        img_second_flip = tf.case([(tf.greater(random_values[2], 0.5), keep_branch)], default=kill_branch)
        # If its less than or equal to 0.5, keep bev
        bev_second_flip = tf.case([(tf.less_equal(random_values[2], 0.5), keep_branch)],
                                  default=kill_branch)

        # Use lambda since this returns another condition and it needs to
        # be callable
        final_img_mask = tf.case([(tf.equal(third_flip, 1), lambda: img_chances)], default=lambda: img_second_flip)

        final_bev_mask = tf.case([(tf.equal(third_flip, 1), lambda: bev_chances)], default=lambda: bev_second_flip)

        return final_img_mask, final_bev_mask
