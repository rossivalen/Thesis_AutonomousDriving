# import cv2
import numpy as np
import os

from PIL import Image

from wavedata.tools.obj_detection import evaluation
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.core.voxel_grid_2d import VoxelGrid2D
from wavedata.tools.core.voxel_grid import VoxelGrid

import avod

from avod.core import box_3d_encoder, anchor_projector
from avod.core import anchor_encoder
from avod.core import anchor_filter
from avod.core.anchor_generators import grid_anchor_3d_generator
import avod.builders.config_builder_util as config_build

from avod.core.label_cluster_utils import LabelClusterUtils

class MiniBatchPreprocessor(object):
    def __init__(self,
                 dataset,
                 mini_batch_dir,
                 anchor_strides,
                 density_threshold,
                 neg_iou_3d_range,
                 pos_iou_3d_range):
        """Preprocesses anchors and saves info to files for RPN training

        Args:
            dataset: Dataset object
            mini_batch_dir: directory to save the info
            anchor_strides: anchor strides for generating anchors (per class)
            density_threshold: minimum number of points required to keep an
                anchor
            neg_iou_3d_range: 3D iou range for an anchor to be negative
            pos_iou_3d_range: 3D iou range for an anchor to be positive
        """

        self.dataset = dataset

        self.mini_batch_dir = avod.root_dir() + '/data/mini_batches/' + \
            'iou_2d/' +  "nuscenes" + '/' + "train" + '/' + "lidar"

        config_path = 'avod/configs/unittest_pipeline.config'
        pipeline_config=config_build.get_configs_from_pipeline_file(config_path, "val")
        self.config = pipeline_config[3].kitti_utils_config
        self.area_extents = np.reshape(self.config.area_extents, (3, 2))
        self._anchor_strides = anchor_strides

        self._density_threshold = density_threshold
        self._negative_iou_range = neg_iou_3d_range
        self._positive_iou_range = pos_iou_3d_range

    def _calculate_anchors_info(self,
                                all_anchor_boxes_3d,
                                empty_anchor_filter,
                                gt_labels):
        """Calculates the list of anchor information in the format:
            N x 8 [max_gt_2d_iou, max_gt_3d_iou, (6 x offsets), class_index]
                max_gt_out - highest 3D iou with any ground truth box
                offsets - encoded offsets [dx, dy, dz, d_dimx, d_dimy, d_dimz]
                class_index - the anchor's class as an index
                    (e.g. 0 or 1, for "Background" or "Car")

        Args:
            all_anchor_boxes_3d: list of anchors in box_3d format
                N x [x, y, z, l, w, h, ry]
            empty_anchor_filter: boolean mask of which anchors are non empty
            gt_labels: list of Object Label data format containing ground truth
                labels to generate positives/negatives from.

        Returns:
            list of anchor info
        """
        # Check for ground truth objects
        if len(gt_labels) == 0:
            raise Warning("No valid ground truth label to generate anchors.")

        # Filter empty anchors
        anchor_indices = np.where(empty_anchor_filter)[0]
        anchor_boxes_3d = all_anchor_boxes_3d[empty_anchor_filter]

        # Convert anchor_boxes_3d to anchor format
        anchors = box_3d_encoder.box_3d_to_anchor(anchor_boxes_3d)

        # Convert gt to boxes_3d -> anchors -> iou format
        gt_boxes_3d = np.asarray(
            [box_3d_encoder.object_label_to_box_3d(gt_obj)
             for gt_obj in gt_labels])
        gt_anchors = box_3d_encoder.box_3d_to_anchor(gt_boxes_3d,
                                                     ortho_rotate=True)

        rpn_iou_type = self.mini_batch_utils.rpn_iou_type
        if rpn_iou_type == '2d':
            # Convert anchors to 2d iou format
            anchors_for_2d_iou, _ = np.asarray(anchor_projector.project_to_bev( anchors, kitti_utils.bev_extents))

            gt_boxes_for_2d_iou, _ = anchor_projector.project_to_bev(gt_anchors, kitti_utils.bev_extents)

        elif rpn_iou_type == '3d':
            # Convert anchors to 3d iou format for calculation
            anchors_for_3d_iou = box_3d_encoder.box_3d_to_3d_iou_format(anchor_boxes_3d)

            gt_boxes_for_3d_iou = \
                box_3d_encoder.box_3d_to_3d_iou_format(gt_boxes_3d)
        else:
            raise ValueError('Invalid rpn_iou_type {}', rpn_iou_type)

        # Initialize sample and offset lists
        num_anchors = len(anchor_boxes_3d)
        all_info = np.zeros((num_anchors, self.mini_batch_utils.col_length))

        # Update anchor indices
        all_info[:, self.mini_batch_utils.col_anchor_indices] = anchor_indices

        # For each of the labels, generate samples
        for gt_idx in range(len(gt_labels)):

            gt_obj = gt_labels[gt_idx]
            gt_box_3d = gt_boxes_3d[gt_idx]

            # Get 2D or 3D IoU for every anchor
            if self.mini_batch_utils.rpn_iou_type == '2d':
                gt_box_for_2d_iou = gt_boxes_for_2d_iou[gt_idx]
                ious = evaluation.two_d_iou(gt_box_for_2d_iou, anchors_for_2d_iou)
            elif self.mini_batch_utils.rpn_iou_type == '3d':
                gt_box_for_3d_iou = gt_boxes_for_3d_iou[gt_idx]
                ious = evaluation.three_d_iou(gt_box_for_3d_iou, anchors_for_3d_iou)

            # Only update indices with a higher iou than before
            update_indices = np.greater( ious, all_info[:, self.mini_batch_utils.col_ious])

            # Get ious to update
            ious_to_update = ious[update_indices]

            # Calculate offsets, use 3D iou to get highest iou
            anchors_to_update = anchors[update_indices]
            gt_anchor = box_3d_encoder.box_3d_to_anchor(gt_box_3d, ortho_rotate=True)
            offsets = anchor_encoder.anchor_to_offset(anchors_to_update, gt_anchor)

            # Convert gt type to index
            class_idx = kitti_utils.class_str_to_index(gt_obj.type)

            # Update anchors info (indices already updated)
            # [index, iou, (offsets), class_index]
            all_info[update_indices, self.mini_batch_utils.col_ious] = ious_to_update

            all_info[update_indices, self.mini_batch_utils.col_offsets_lo: self.mini_batch_utils.col_offsets_hi] = offsets
            all_info[update_indices, self.mini_batch_utils.col_class_idx] = class_idx

        return all_info

    def preprocess(self, indices):
        """Preprocesses anchor info and saves info to files

        Args:
            indices (int array): sample indices to process.
                If None, processes all samples
        """
        # Get anchor stride for class
        anchor_strides = self._anchor_strides
        dataset = self.dataset
        classes_name = ["car", "pedestrian", "bus"]

        # Make folder if it doesn't exist yet
        for i in classes_name:
            output_dir = self.get_file_path(i, anchor_strides, sample_name=None)
            os.makedirs(output_dir, exist_ok=True)

        # Get clusters for class
        label_cluster_utils = LabelClusterUtils(dataset)
        all_clusters_sizes, _ = label_cluster_utils.get_clusters(5, dataset)

        anchor_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()     #check

        # Load indices of data_split
        all_samples = label_cluster_utils.sample_list

        if indices is None:
            indices = np.array(all_samples)
        num_samples = len(indices)

        # For each image in the dataset, save info on the anchors
        for sample_idx in indices:
            # recall that sample_idx is the token that gets you the sample_data
        
            # Check for existing files and skip to the next
            if self._check_for_existing(classes_name, anchor_strides, sample_idx):
                print("{} / {}: Sample already preprocessed".format(sample_idx, num_samples, sample_name))
                continue

            # Get ground truth and filter based on difficulty
            ground_truth_list = preproc_helper.read_labels(label_cluster_utils.label_dir, img_idx)

            # Filter objects to dataset classes
            filtered_gt_list = self.filter_labels(ground_truth_list)
            filtered_gt_list = np.asarray(filtered_gt_list)

            # Filtering by class has no valid ground truth, skip this image
            if len(filtered_gt_list) == 0:
                print("{} / {} No {}s for sample {} "
                      "(Ground Truth Filter)".format(sample_idx, num_samples, classes_name, sample_name))

                # Output an empty file and move on to the next image.
                self._save_to_file(classes_name, anchor_strides, sample_name)
                continue

            # Get ground plane maybe move all the gets in getroadplane
            sample_data = dataset.get("sample", sample_idx)
            
            cam_front_token = dataset.get('sample_data', sample_data["data"]["CAM_FRONT"])
            cam_front_data = cam_front_token.get("calibrated_sensor_token")
            cam_front_calib = dataset.get("calibrated_sensor", cam_front_data )
            cam_front_coords = cam_front_calib.get("translation")
            
            cam_front_right_token = dataset.get('sample_data', sample_data["data"]["CAM_FRONT_RIGHT"])
            cam_front_right_data = cam_front_right_token.get("calibrated_sensor_token")
            cam_front_right_calib = dataset.get("calibrated_sensor", cam_front_right_data )
            cam_front_right_coords = cam_front_right_calib.get("translation")
            
            cam_front_left_token = dataset.get('sample_data', sample_data["data"]["CAM_LEFT_RIGHT"])
            cam_front_left_data = cam_front_left_token.get("calibrated_sensor_token")
            cam_front_left_calib = dataset.get("calibrated_sensor", cam_front_left_data )
            cam_front_left_coords = cam_front_left_calib.get("translation")
            
            ground_plane = frame_helper.get_ground_plane_coefficients(cam_front_coords, cam_front_left_coords, cam_front_right_coords)
            
            image = Image.open(file_name=dataset.get_sample_data_path(cam_front_token))
            image_shape = [image.size[1], image.size[0]]

            # Generate sliced 2D voxel grid for filtering HERE 
            sample_lidar_name = sample_data["data"]["LIDAR_TOP"]
            vx_grid_2d = self.create_sliced_voxel_grid_2d(sample_lidar_name, ground_plane, dataset, 
                                                                   source="lidar", image_shape=image_shape)

            # List for merging all anchors
            all_anchor_boxes_3d = []

            # Create anchors for each class
            for class_idx in range(len(dataset.classes)):
                # Generate anchors for all classes
                grid_anchor_boxes_3d = anchor_generator.generate(
                    area_3d=self._area_extents,
                    anchor_3d_sizes=all_clusters_sizes[class_idx],
                    anchor_stride=self._anchor_strides[class_idx],
                    ground_plane=ground_plane)

                all_anchor_boxes_3d.extend(grid_anchor_boxes_3d)

            # Filter empty anchors
            all_anchor_boxes_3d = np.asarray(all_anchor_boxes_3d)
            anchors = box_3d_encoder.box_3d_to_anchor(all_anchor_boxes_3d)
            empty_anchor_filter = anchor_filter.get_empty_anchor_filter_2d(anchors, vx_grid_2d, self._density_threshold)

            # Calculate anchor info
            anchors_info = self._calculate_anchors_info( all_anchor_boxes_3d, empty_anchor_filter, filtered_gt_list)

            anchor_ious = anchors_info[:, self.mini_batch_utils.col_ious]

            valid_iou_indices = np.where(anchor_ious > 0.0)[0]

            print("{} / {}:"
                  "{:>6} anchors, "
                  "{:>6} iou > 0.0, "
                  "for {:>3} {}(s) for sample {}".format(
                      sample_idx + 1, num_samples,
                      len(anchors_info),
                      len(valid_iou_indices),
                      len(filtered_gt_list), classes_name, sample_name ))

            # Save anchors info
            self._save_to_file(classes_name, anchor_strides, sample_name, anchors_info)

    def _check_for_existing(self, classes_name, anchor_strides, sample_name):
        """
        Checks if a mini batch file exists already

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            anchor_strides: anchor strides
            sample_name (str): sample name from dataset, e.g. '000123'

        Returns:
            True if the anchors info file already exists
        """

        file_name = self.mini_batch_utils.get_file_path(classes_name,
                                                        anchor_strides,
                                                        sample_name)
        if os.path.exists(file_name):
            return True

        return False

    def _save_to_file(self, classes_name, anchor_strides, sample_name,
                      anchors_info=np.array([])):
        """
        Saves the anchors info matrix to a file

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            anchor_strides: anchor strides
            sample_name (str): name of sample, e.g. '000123'
            anchors_info: ndarray of anchor info of shape (N, 8)
                N x [index, iou, (6 x offsets), class_index], defaults to
                an empty array
        """

        file_name = self.mini_batch_utils.get_file_path(classes_name,
                                                        anchor_strides,
                                                        sample_name)

        # Save to npy file
        anchors_info = np.asarray(anchors_info, dtype=np.float32)
        np.save(file_name, anchors_info)

    def create_sliced_voxel_grid_2d(self, sample_lidar_name, ground_plane, dataset, source, image_shape=None):
        """Generates a filtered 2D voxel grid from point cloud data

        Args:
            sample_name: image name to generate stereo pointcloud from
            source: point cloud source, e.g. 'lidar'
            image_shape: image dimensions [h, w], only required when
                source is 'lidar' or 'depth'

        Returns:
            voxel_grid_2d: 3d voxel grid from the given image
        """
        img_idx = int(sample_name)
        
        lidar_data = dataset.get("sample_data", sample_lidar_token)
        lidar_filepath = dataset.get_sample_data_path(sample_lidar_token)
        pointcloud = LidarPointCloud.from_file(lidar_filepath)
        
        #MOVE THE FUNCTIONS FROM OBJ UTILS TO FRAME_HELPER
        filtered_points = self._apply_slice_filter(point_cloud, ground_plane)

        # Create Voxel Grid 
        voxel_grid_2d = VoxelGrid2D()
        voxel_grid_2d.voxelize_2d(filtered_points, self.voxel_size,
                                  extents=self.area_extents,
                                  ground_plane=ground_plane,
                                  create_leaf_layout=True)

        return voxel_grid_2d
    
    def _apply_slice_filter(self, point_cloud, ground_plane,
                            height_lo=0.2, height_hi=2.0):
        """ Applies a slice filter to the point cloud

        Args:
            point_cloud: A point cloud in the shape (3, N)
            ground_plane: ground plane coefficients
            height_lo: (optional) lower height for slicing
            height_hi: (optional) upper height for slicing

        Returns:
            Points filtered with a slice filter in the shape (N, 3)
        """

        slice_filter = self.create_slice_filter(point_cloud,
                                                self.area_extents,
                                                ground_plane,
                                                height_lo, height_hi)

        # Transpose point cloud into N x 3 points
        points = np.asarray(point_cloud).T

        filtered_points = points[slice_filter]

        return filtered_points
    def create_slice_filter(self, point_cloud, area_extents,
                            ground_plane, ground_offset_dist, offset_dist):
        """ Creates a slice filter to take a slice of the point cloud between
            ground_offset_dist and offset_dist above the ground plane

        Args:
            point_cloud: Point cloud in the shape (3, N)
            area_extents: 3D area extents
            ground_plane: ground plane coefficients
            offset_dist: max distance above the ground
            ground_offset_dist: min distance above the ground plane

        Returns:
            A boolean mask if shape (N,) where
                True indicates the point should be kept
                False indicates the point should be removed
        """
        #MOVE THESE GETPOINT INTO YOUR FRAME_HELPER
        # Filter points within certain xyz range and offset from ground plane
        offset_filter = obj_utils.get_point_filter(point_cloud, area_extents,
                                                   ground_plane, offset_dist)

        # Filter points within 0.2m of the road plane
        road_filter = obj_utils.get_point_filter(point_cloud, area_extents,
                                                 ground_plane,
                                                 ground_offset_dist)

        slice_filter = np.logical_xor(offset_filter, road_filter)
        return slice_filter
    
    def filter_labels(self, objects,
                      classes=None,
                      difficulty=None,
                      max_occlusion=None):
        """Filters ground truth labels based on class, difficulty, and
        maximum occlusion

        Args:
            objects: A list of ground truth instances of Object Label
            classes: (optional) classes to filter by, if None
                all classes are used
            difficulty: (optional) KITTI difficulty rating as integer
            max_occlusion: (optional) maximum occlusion to filter objects

        Returns:
            filtered object label list
        """
        if classes is None:
            cat = self.dataset.category
            classes=[]
            for i in cat:
                classes.append(i.get("name"))

        objects = np.asanyarray(objects)
        filter_mask = np.ones(len(objects), dtype=np.bool)

        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]

            if filter_mask[obj_idx]:
                if not self._check_class(obj, classes):
                    filter_mask[obj_idx] = False
                    continue
                    
        return objects[filter_mask]
    
    def _check_class(self, obj, classes):
        """This filters an object by class.
        Args:
            obj: An instance of ground-truth Object Label
        Returns: True or False depending on whether the object
            matches the desired class.
        """
        return obj.cls in classes
    
    def get_file_path(self, classes_name, anchor_strides, sample_name):
        """Gets the full file path to the anchors info

        Args:
            classes_name: name of classes ('Car', 'Pedestrian', 'Cyclist',
                'People')
            anchor_strides: anchor strides
            sample_name: sample name, e.g. '000123'

        Returns:
            The anchors info file path. Returns the folder if
                sample_name is None
        """
        # Round values for nicer folder names
        anchor_strides = np.round(anchor_strides[:, 0], 3)

        anchor_strides_str = ' '.join(str(stride) for stride in anchor_strides)

        return self.mini_batch_dir + '/' + classes_name + \
            '[ ' + "0.5" + ']'
