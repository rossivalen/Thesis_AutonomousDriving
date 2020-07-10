import numpy as np
import os
from pyquaternion import Quaternion

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
        self.ry = (0., 0., 0., 0.)
        self.score = 0.

def read_labels(label_dir, sample_token, dataset, results=False):
    """Reads in label data file from Kitti Dataset.

    Returns:
    obj_list -- List of instances of class ObjectLabel.

    Keyword arguments:
    label_dir -- directory of the label files
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
        sample_annot = dataset.get("sample_annotation", tokens[0])
        obj.cls = sample_annot.get("category_name")
        obj.visibility = sample_annot.get("visibility_token")
        size = sample_annot.get("size")
        obj.h = size[2]
        obj.w = size[0]
        obj.l = size[1]
        obj.t = sample_annot.get("translation")
        rotation = sample_annot.get("rotation")
        quaternion_ry = Quaternion(rotation)
        obj.ry = sample_annot.get("rotation")
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