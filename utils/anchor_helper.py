import numpy as np
import tensorflow as tf

from wavedata.tools.core.voxel_grid_2d import VoxelGrid2D
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

import avod.core.format_checker as fc
from avod.core.minibatch_samplers import balanced_positive_negative_sampler

def sample_mini_batch(max_ious,
                          mini_batch_size,
                          negative_iou_range,
                          positive_iou_range):
        """
        Samples a mini batch based on anchor ious with ground truth

        Args:
            max_ious: a tensor of max ious with ground truth in
                the shape (N,)
            mini_batch_size: size of the mini batch to return
            negative_iou_range: iou range to consider an anchor as negative
            positive_iou_range: iou range to consider an anchor as positive

        Returns:
            mb_sampled: a boolean mask where True indicates anchors sampled
                for the mini batch
            mb_pos_sampled: a boolean mask where True indicates positive anchors
        """

        bkg_and_neg_labels = tf.less(max_ious, negative_iou_range[1])
        pos_labels = tf.greater(max_ious, positive_iou_range[0])
        indicator = tf.logical_or(pos_labels, bkg_and_neg_labels)

        if negative_iou_range[0] > 0.0:
            # If neg_iou_lo is > 0.0, the mini batch may be empty.
            # In that case, use all background and negative labels
            neg_labels = tf.logical_and(bkg_and_neg_labels, tf.greater_equal(max_ious, negative_iou_range[0]))

            new_indicator = tf.logical_or(pos_labels, neg_labels)

            num_valid = tf.reduce_sum(tf.cast(indicator, tf.int32))
            indicator = tf.cond(tf.greater(num_valid, 0), true_fn=lambda: tf.identity(new_indicator), 
                                false_fn=lambda: tf.identity(bkg_and_neg_labels))

        sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler()
        mb_sampled, mb_pos_sampled = sampler.subsample( indicator, mini_batch_size, pos_labels)

        return mb_sampled, mb_pos_sampled
    
def anchor_to_offset(anchors, ground_truth):
    """Encodes the anchor regression predictions with the
    ground truth.

    Args:
        anchors: A numpy array of shape (N, 6) representing
            the generated anchors.
        ground_truth: A numpy array of shape (6,) containing
            the label boxes in the anchor format.

    Returns:
        anchor_offsets: A numpy array of shape (N, 6)
            encoded/normalized with the ground-truth, representing the
            offsets.
    """

    fc.check_anchor_format(anchors)

    anchors = np.asarray(anchors).reshape(-1, 6)
    ground_truth = np.reshape(ground_truth, (6,))

    # t_x_gt = (x_gt - x_anch)/dim_x_anch
    t_x_gt = (ground_truth[0] - anchors[:, 0]) / anchors[:, 3]
    # t_y_gt = (y_gt - y_anch)/dim_y_anch
    t_y_gt = (ground_truth[1] - anchors[:, 1]) / anchors[:, 4]
    # t_z_gt = (z_gt - z_anch)/dim_z_anch
    t_z_gt = (ground_truth[2] - anchors[:, 2]) / anchors[:, 5]
    # t_dx_gt = log(dim_x_gt/dim_x_anch)
    t_dx_gt = np.log(ground_truth[3] / anchors[:, 3])
    # t_dy_gt = log(dim_y_gt/dim_y_anch)
    t_dy_gt = np.log(ground_truth[4] / anchors[:, 4])
    # t_dz_gt = log(dim_z_gt/dim_z_anch)
    t_dz_gt = np.log(ground_truth[5] / anchors[:, 5])
    anchor_offsets = np.stack((t_x_gt, t_y_gt, t_z_gt, t_dx_gt, t_dy_gt, t_dz_gt), axis=1)
    return anchor_offsets


def tf_anchor_to_offset(anchors, ground_truth):
    """Encodes the anchor regression predictions with the
    ground truth.

    This function assumes the ground_truth tensor has been arranged
    in a way that each corresponding row in ground_truth, is matched
    with that anchor according to the highest IoU.
    For instance, the ground_truth might be a matrix of shape (256, 6)
    of repeated entries for the original ground truth of shape (x, 6),
    where each entry has been selected as the highest IoU match with that
    anchor. This is different from the same function in numpy format, where
    we loop through all the ground truth anchors, and calculate IoUs for
    each and then select the match with the highest IoU.

    Args:
        anchors: A tensor of shape (N, 6) representing
            the generated anchors.
        ground_truth: A tensor of shape (N, 6) containing
            the label boxes in the anchor format. Each ground-truth entry
            has been matched with the anchor in the same entry as having
            the highest IoU.

    Returns:
        anchor_offsets: A tensor of shape (N, 6)
            encoded/normalized with the ground-truth, representing the
            offsets.
    """

    fc.check_anchor_format(anchors)

    # Make sure anchors and anchor_gts have the same shape
    dim_cond = tf.equal(tf.shape(anchors), tf.shape(ground_truth))

    with tf.control_dependencies([dim_cond]):
        t_x_gt = (ground_truth[:, 0] - anchors[:, 0]) / anchors[:, 3]
        t_y_gt = (ground_truth[:, 1] - anchors[:, 1]) / anchors[:, 4]
        t_z_gt = (ground_truth[:, 2] - anchors[:, 2]) / anchors[:, 5]
        t_dx_gt = tf.math.log(ground_truth[:, 3] / anchors[:, 3])
        t_dy_gt = tf.math.log(ground_truth[:, 4] / anchors[:, 4])
        t_dz_gt = tf.math.log(ground_truth[:, 5] / anchors[:, 5])
        anchor_offsets = tf.stack((t_x_gt,
                                   t_y_gt,
                                   t_z_gt,
                                   t_dx_gt,
                                   t_dy_gt,
                                   t_dz_gt), axis=1)

        return anchor_offsets


def offset_to_anchor(anchors, offsets):
    """Decodes the anchor regression predictions with the
    anchor.

    Args:
        anchors: A numpy array or a tensor of shape [N, 6]
            representing the generated anchors.
        offsets: A numpy array or a tensor of shape
            [N, 6] containing the predicted offsets in the
            anchor format  [x, y, z, dim_x, dim_y, dim_z].

    Returns:
        anchors: A numpy array of shape [N, 6]
            representing the predicted anchor boxes.
    """

    fc.check_anchor_format(anchors)
    fc.check_anchor_format(offsets)

    # x = dx * dim_x + x_anch
    x_pred = (offsets[:, 0] * anchors[:, 3]) + anchors[:, 0]
    # y = dy * dim_y + x_anch
    y_pred = (offsets[:, 1] * anchors[:, 4]) + anchors[:, 1]
    # z = dz * dim_z + z_anch
    z_pred = (offsets[:, 2] * anchors[:, 5]) + anchors[:, 2]

    tensor_format = isinstance(anchors, tf.Tensor)
    if tensor_format:
        # dim_x = exp(log(dim_x) + dx)
        dx_pred = tf.math.exp(tf.math.log(anchors[:, 3]) + offsets[:, 3])
        # dim_y = exp(log(dim_y) + dy)
        dy_pred = tf.math.exp(tf.math.log(anchors[:, 4]) + offsets[:, 4])
        # dim_z = exp(log(dim_z) + dz)
        dz_pred = tf.math.exp(tf.math.log(anchors[:, 5]) + offsets[:, 5])
        anchors = tf.stack((x_pred,
                            y_pred,
                            z_pred,
                            dx_pred,
                            dy_pred,
                            dz_pred), axis=1)
    else:
        dx_pred = np.exp(np.log(anchors[:, 3]) + offsets[:, 3])
        dy_pred = np.exp(np.log(anchors[:, 4]) + offsets[:, 4])
        dz_pred = np.exp(np.log(anchors[:, 5]) + offsets[:, 5])
        anchors = np.stack((x_pred,
                            y_pred,
                            z_pred,
                            dx_pred,
                            dy_pred,
                            dz_pred), axis=1)

    return anchors

def create_sliced_voxel_grid_2d(self, dataset, sample_token, source, ground_plane,
                                image_shape=None):
    """Generates a filtered 2D voxel grid from point cloud data

    Args:
        sample_token: sample lidar token to get the pointcloud
        source: point cloud source, e.g. 'lidar'
        image_shape: image dimensions [h, w], only required when
            source is 'lidar' or 'depth'

    Returns:
        voxel_grid_2d: 3d voxel grid from the given image
    """
    img_idx = int(sample_name)
    lidar_filepath = dataset.get_sample_data_path(sample_token)
    lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
    filtered_points = self._apply_slice_filter(point_cloud, ground_plane)

    # Create Voxel Grid
    voxel_grid_2d = VoxelGrid2D()
    area_extents = np.reshape([-40, 40, -5, 3, 0, 70], (3, 2))
    voxel_grid_2d.voxelize_2d(filtered_points, 0.1, extents=area_extents, ground_plane=ground_plane, create_leaf_layout=True)

    return voxel_grid_2d