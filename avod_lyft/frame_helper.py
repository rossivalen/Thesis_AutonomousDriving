import csv
import numpy as np
import cv2
import os


class FrameCalibrationData:
    """Frame Calibration Holder
        3x3    intrinsic   Intrinsic camera matrix.

        3x3    rotation    Rotation matrix.
        
        3x1    translation Translation vector

        """

    def __init__(self):
        self.intrinsic = []
        self.rotation = []
        self.translation = []

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
    rotation_temp = calib_sensor_data.get("rotation")
    translation_temp = calib_sensor_data.get("translation")
    
    frame_calibration_info=FrameCalibrationData()
    frame_calibration_info.intrinsic=intrinsic_temp
    frame_calibration_info.rotation=rotation_temp
    frame_calibration_info.translation=translation_temp
    
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