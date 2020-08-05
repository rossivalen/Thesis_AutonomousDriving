import numpy as np
import os
from pyquaternion import Quaternion
import math
from avod.core import box_3d_encoder
import avod.builders.config_builder_util as config_build
from avod.core.mini_batch_utils import MiniBatchUtils

class ObjectLabel:
    """Object Label Class
    1    cls           Describes the type of object: 'Car', 'Van', 'Truck',
                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                      'Misc' or 'DontCare'

    1    visibility   If no annotation, left empty

    3    dimensions   3D object dimensions: height, width, length (in meters)

    3    location     3D object location of the center x,y,z in camera coordinates (in meters)

    4    rotation_y   Rotation quaternion

    1    score        Only for results: Float, indicating confidence in
                      detection, needed for p/r curves, higher is better.
    """

    def __init__(self):
        self.cls = ""  # Type of object
        self.visibility = 0.
        self.h = 0.
        self.w = 0.
        self.l = 0.
        self.t = (0., 0., 0.)
        self.ry = 0.
        self.score = 0.
        
class Config:
    
    def __init__(self):
        config_path = 'avod/configs/unittest_pipeline.config'
        pipeline_config=config_build.get_configs_from_pipeline_file(config_path, "val")
        self.config = pipeline_config[3].kitti_utils_config
        self.scene_idx = 5
        self.area_extents = np.reshape(self.config.area_extents, (3, 2))
        self.bev_extents = self.area_extents[[0, 2]]
        self.voxel_size = self.config.voxel_size
        self.anchor_strides = np.reshape(self.config.anchor_strides, (-1, 2))

def read_labels(sample_token, dataset, results=False):
    """Reads in label data.

    Returns:
    obj_list -- List of instances of class ObjectLabel.

    Keyword arguments:
    sample_token -- sample token you want to analyze
    """
    # To see which token it gets, look at load labels!
    
    # Define the object list
    obj_list = []
    
    sample_data = dataset.get("sample", sample_token)
    tokens=sample_data['anns']
        
    for i in range(len(tokens)):
        obj = ObjectLabel()
        token = tokens[i]
        sample_annot = dataset.get("sample_annotation", token)
        obj.cls = sample_annot.get("category_name")
        obj.visibility = sample_annot.get("visibility_token")
        size = sample_annot.get("size")
        obj.h = size[2]
        obj.w = size[0]
        obj.l = size[1]
        obj.t = sample_annot.get("translation")
        rotation = sample_annot.get("rotation")
        quaternion = Quaternion(rotation)
        X,Y,Z = quaternion_to_euler( quaternion )
        ry= math.radians(Y)
        obj.ry = ry
#       still have to write the score class
#       if results:
#           obj.score = float(p[idx, 15])
#       else:
        obj.score = 0.0
        obj_list.append(obj)

    return obj_list

def load_sample_names(scene_idx, dataset, all_scenes=False):
    """Load the sample names of a selected scene or of all scenes.

    Args:
    scene_idx: scene index to be analyzed.
    all_scenes: if True, analyze all scenes, starting from scene_idx to end
    
    Returns:
    A list of sample names (file names)
    """
    if all_scenes==False:
        token_list=[]
                                                    
        my_scene = dataset.scene[scene_idx] 
        my_sample_token = my_scene["first_sample_token"]
        my_sample = dataset.get('sample', my_sample_token)
        my_sample_data = dataset.get('sample_data', my_sample['data']["CAM_FRONT"])
        tok = my_sample_data.get("sample_token")
        token_list.append(tok)
        
        my_last_sample_token = my_scene["last_sample_token"]
                                                    
        while my_sample_token != my_last_sample_token:
            my_sample_token = dataset.get("sample", my_sample_token)["next"]
            my_sample = dataset.get('sample', my_sample_token)
            my_sample_data = dataset.get('sample_data', my_sample['data']["CAM_FRONT"])
            tok = my_sample_data.get("sample_token")
            token_list.append(tok)  
                                                    
        return token_list
    else:
        for i in range(scene_idx, 148):
            token_list=[]

            my_scene = dataset.scene[i] 
            my_sample_token = my_scene["first_sample_token"]
            my_sample = dataset.get('sample', my_sample_token)
            my_sample_data = dataset.get('sample_data', my_sample['data']["CAM_FRONT"])
            tok = my_sample_data.get("sample_token")
            token_list.append(tok)

            my_last_sample_token = my_scene["last_sample_token"]

            while my_sample_token != my_last_sample_token:
                my_sample_token = dataset.get("sample", my_sample_token)["next"]
                my_sample = dataset.get('sample', my_sample_token)
                my_sample_data = dataset.get('sample_data', my_sample['data']["CAM_FRONT"])
                tok = my_sample_data.get("sample_token")
                token_list.append(tok)  
                                                    
        return token_list
def quaternion_to_euler( quaternion ):

    x=quaternion[0]
    y=quaternion[1]
    z=quaternion[2]
    w=quaternion[3]
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

def load_sample_info(sample_name, classes, dataset):
    
    anchors_info = get_anchors_info(dataset, classes, sample_name)
    obj_labels = read_labels(sample_name, dataset, results=False)
    train_on_all_samples = False
    
    # Only use objects that match dataset classes
    if obj_labels is not None:
            label_boxes_3d = np.asarray([box_3d_encoder.object_label_to_box_3d(obj_label) for obj_label in obj_labels])
            obj_classes = [obj_label.cls for obj_label in obj_labels]
            label_classes = [class_str_to_index(obj_label.cls) for obj_label in obj_labels]
            label_classes = np.asarray(label_classes, dtype=np.int32)
            # Return empty anchors_info if no ground truth after filtering
            if len(label_boxes_3d) == 0:
                anchors_info = []
                if train_on_all_samples:
                    # If training without any positive labels, we cannot
                    # set these to zeros, because later on the offset calc
                    # uses log on these anchors. So setting any arbitrary
                    # number here that does not break the offset calculation
                    # should work, since the negative samples won't be
                    # regressed in any case.
                    dummy_anchors = [[-1000, -1000, -1000, 1, 1, 1]]
                    label_anchors = np.asarray(dummy_anchors)
                    dummy_boxes = [[-1000, -1000, -1000, 1, 1, 1, 0]]
                    label_boxes_3d = np.asarray(dummy_boxes)
                else:
                    label_anchors = np.zeros((1, 6))
                    label_boxes_3d = np.zeros((1, 7))
                    label_classes = np.zeros(1)
            else:
                label_anchors = box_3d_encoder.box_3d_to_anchor(label_boxes_3d, ortho_rotate=True)
                
    return anchors_info, obj_classes, label_classes, label_anchors, label_boxes_3d
    
def get_anchors_info(dataset, classes_name, sample_name):
    
        config = Config()
        mini_batch_utils = MiniBatchUtils(dataset)
        anchors_info = mini_batch_utils.get_anchors_info(classes_name, config.anchor_strides, sample_name)
        return anchors_info
def class_str_to_index(class_str):
        """
        Converts an object class type string into a integer index

        Args:
            class_str: the object type (e.g. 'Car', 'Pedestrian', or 'Cyclist')

        Returns:
            The corresponding integer index for a class type, starting at 1
            (0 is reserved for the background class).
            Returns -1 if we don't care about that class type.
        """
        clss=["car", "pedestrian", "bus"]
        if class_str in clss:
            return clss.index(class_str) + 1

    