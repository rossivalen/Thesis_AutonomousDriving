import csv
import numpy as np
import cv2
import os
from pyquaternion import Quaternion

class FrameCalibrationData:
    """Frame Calibration Holder
        3x3    intrinsic     Intrinsic camera matrix.

        3x3    rotation      Rotation matrix.
        
        3x1    translation   Translation vector
        
        3x4    cam_matrix    Camera matrix

        """

    def __init__(self):
        self.intrinsic = []
        self.rotation = []
        self.translation = []
        self.cam_matrix = []

def read_calibration(sensor_token, dataset):
    """Reads in Calibration file from Kitti Dataset.

    Keyword Arguments:
    ------------------
    sensor_token: keyword argument of the sensor that captured the frame.

    Returns:
    --------
    frame_calibration_info : frame full calibration info

    """
    
    calib_sensor_data = dataset.get("calibrated_sensor", sensor_token)
    intrinsic_temp = calib_sensor_data.get("camera_intrinsic")
    rotation_temp = Quaternion(calib_sensor_data.get("rotation"))
    rot_m = rotation_temp.rotation_matrix 
    translation_temp = calib_sensor_data.get("translation")
    
    R=np.matrix([rot_m[0], rot_m[1], rot_m[2], translation_temp])
    intrinsic=np.matrix(intrinsic_temp)
    X=np.matmul(intrinsic, R.T)
    
    frame_calibration_info=FrameCalibrationData()
    frame_calibration_info.intrinsic=intrinsic
    frame_calibration_info.rotation=rot_m
    frame_calibration_info.translation=translation_temp
    frame_calibration_info.cam_matrix=X
    
    return frame_calibration_info

def get_ground_plane_coeff(point1, point2, point3):
    """
    Get ground plane coeffients under the hypothesis that cameras are mounted 
    approximately at ground plane level (as in KITTI). 
    
    :param point1: x,y,z coordintes of the CAM_FRONT
    :param point2: x,y,z coordintes of the CAM_FRONT_RIGHT
    :param point3: x,y,z coordintes of the CAM_FRONT_LEFT
    
    :returns: a,b,c,d coefficients of the ground plane
    """
    p1 = np.asarray(point1)
    p2 = np.asarray(point2)
    p3 = np.asarray(point3)
    
    v1 = p3-p1
    v2 = p2-p1
    cp = np.cross(v1, v2)
    a,b,c = cp
    d = np.dot(cp, p3)
    return a,b,c,d

def get_cam_matrix(intrinsic, rotation, translation):
    """
    Get 3x4 camera matrix.
    
    :param intrinsic:   intrinsic matrix 3x3
    :param rotation:    rotation matrix 3x3
    :param translation: translation vetor 3x1
    
    :returns: 3x4 camera matrix
    
    """
    intrinsic=np.matrix(intrinsic)
    R=np.matrix([rotation[0], rotation[1], rotation[2], translation])
    return np.matmul(intrinsic, R.T)

def project_to_image(point_cloud, p):
    """ Projects a 3D point cloud to 2D points for plotting

    :param point_cloud: 3D point cloud (3, N)
    :param p: Camera matrix (3, 4)

    :return: pts_2d: the image coordinates of the 3D points in the shape (2, N)
    """

    pts_2d = np.dot(p, np.append(point_cloud,
                                 np.ones((1, point_cloud.shape[1])),
                                 axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d

def get_point_filter(point_cloud, extents, ground_plane=None, offset_dist=2.0):
    """
    Creates a point filter using the 3D extents and ground plane

    :param point_cloud: Point cloud in the form [[x,...],[y,...],[z,...]]
    :param extents: 3D area in the form
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    :param ground_plane: Optional, coefficients of the ground plane
        (a, b, c, d)
    :param offset_dist: If ground_plane is provided, removes points above
        this offset from the ground_plane
    :return: A binary mask for points within the extents and offset plane
    """

    point_cloud = np.asarray(point_cloud)

    # Filter points within certain xyz range
    x_extents = extents[0]
    y_extents = extents[1]
    z_extents = extents[2]

    extents_filter = (point_cloud[0] > x_extents[0]) & \
                     (point_cloud[0] < x_extents[1]) & \
                     (point_cloud[1] > y_extents[0]) & \
                     (point_cloud[1] < y_extents[1]) & \
                     (point_cloud[2] > z_extents[0]) & \
                     (point_cloud[2] < z_extents[1])

    if ground_plane is not None:
        ground_plane = np.array(ground_plane)

        # Calculate filter using ground plane
        ones_col = np.ones(point_cloud.shape[1])
        padded_points = np.vstack([point_cloud, ones_col])

        offset_plane = ground_plane + [0, 0, 0, -offset_dist]

        # Create plane filter
        dot_prod = np.dot(offset_plane, padded_points)
        plane_filter = dot_prod < 0

        # Combine the two filters
        point_filter = np.logical_and(extents_filter, plane_filter)
    else:
        # Only use the extents for filtering
        point_filter = extents_filter

    return point_filter