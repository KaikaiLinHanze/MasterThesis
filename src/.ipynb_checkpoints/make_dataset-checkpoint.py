"""
File: make_dataset.py
Name: Kai
----------------------------------------
This file is used to make dataset for training model.
"""
import numpy as np
import json
import os
import shutil,zipfile
import sys
from sklearn.model_selection import train_test_split
from config import *
from data_prepare import data
sys.path.append(AMFTracker_path)
from amftrack.pipeline.functions.image_processing.experiment_util import *
from amftrack.util.geometry import *
from amftrack.pipeline.functions.image_processing.extract_width_fun import *


def get_labelme_segments(directory,direct = True):
    """
    # https://github.com/wkentaro/labelme
    Get the segment from json file (labelme).
    
    directory: str, the directory to the labelme folder.
    direct   : boolen, True for type the absolute path
                       False for only add the folder name within the directory.
                       
    return   : dict, key: position ; value: x,y of the position
    
    Notice   : label_uncertain_check folder do not contain any labelme file. 
               You need to go deeper into the folder inside.
    """
    if direct == 0 :
        directory = labelme_path + directory
    labelme_dict = {}
    def load_labelmefile(file):
        """
        This function is used to load json file.
        
        file: str, the path of the file.
        """
        with open(file) as r:
            return json.load(r)
    for labelmefiles in os.listdir(directory):
        if ".json" in labelmefiles:
            labelme_path = os.path.join(directory,labelmefiles)
            labelmefile = load_labelmefile(labelme_path)
            for shape in labelmefile["shapes"]:
                if shape["shape_type"] == "line":
                    if shape["label"] in labelme_dict.keys():
                        labelme_dict[shape["label"]].append(shape["points"])
                    else:
                        labelme_dict[shape["label"]]= [shape["points"]]
    return labelme_dict

def compute_width_from_segment(segment,camera_res=3.45, magnification=50):
    """
    Convert segment pixel into width.
    
    segment      : dict, data from get_labelme_segments
    camera_res   : int, the resolution of the camera.
    magnification: int, the magnification of the microscope.
    
    return       : dict, key: position ;  value: width
                   dict, key: position ;  value: std od the width
    
    Notice: before change the default value, you must make sure that the value is correct.
    """
    def convert_to_micrometer(pixel_length,camera_res, magnification):
        """
        Converts pixels into micrometers, based on the magnification of the microscope.
        
        pixel_length : int, the length of the segment
        camera_res   : int, the resolution of the camera.
        magnification: int, the magnification of the microscope.
        
        return: int, the width of the segment
        
        Notice: before change the default value, you must make sure that the value is correct.
        """
        return pixel_length * camera_res / magnification
    width_dict = {}
    width_std_dict = {}
    for key in segment.keys():
        widths = []
        for point1,point2 in segment[key]:
            point1 = np.array(point1)
            point2 = np.array(point2)
            width = convert_to_micrometer(np.linalg.norm(point1 - point2), camera_res, magnification)
            widths.append(width)
        width_dict[key] = np.mean(widths)
        width_std_dict[key] = np.std(widths)
    return width_dict, width_std_dict


def get_slices_dict(df,exp,target_length=120,output_type="check"):
    """
    Use the df to collect the position and edge data for checking the feature.
    
    df           : object, df for collect the position and edge information
    exp          : object, Experiment class
    target_length: int, the profile of the target
    output_type  : str, check: for plotting the image of cropped edge for checking whether the edge is correspond the PRINCE data.
                 : str, data : after check every image is correct. you can use it to get the slices.
                 
    return       : dict, key: position ;value: feature
    """
    slices_dict = {}
    f = lambda n: generate_index_along_sequence(n, resolution=4, offset=5)
    f_profiles = lambda edge: extract_section_profiles_for_edge(
        exp, 0, edge, resolution=5, offset=5, step=1,target_length=target_length)
    for index, row in df.iterrows():
        node,node2 = row["Node"] , row["Node2"]
        try:
            edge = get_edge_from_node_labels(exp, 0, node, node2)
            if output_type == "check":
                plot_edge_cropped(edge, 0, mode=3, f=f)
            if output_type == "data":
                slices, coords1, coords2 = f_profiles(edge)
                slices_dict[index] = slices
        except AttributeError:
            print(index, " will not be processed!")
    return slices_dict

def make_dataset(dataset_path,file_name,slice_array,label_array,plate_number:str,strain:str,crossdate:str,prince_position:str,bin_number:str):
    """
    For creating a dataset.
    
    dataset_path    : str, the path that you want to storage the dataset
    file_name       : str, the folder name
    slice_array     : array, the feature
    label_array     : array, the label
    plate_number    : str, the number of the plate
    strain          : str, the strain of the AMF
    crossdate       : str, the crossdate of the AMF
    prince_position : str, the position at the PRINCE
    bin_number      : str, the bin number you set when you extract the data.
    """
    os.mkdir(os.path.join(dataset_path, str(file_name)))
    cv2.imwrite(os.path.join(dataset_path, str(file_name),"slices.png"), slice_array)
    with open(os.path.join(dataset_path, str(file_name),"labels.npy"), "wb") as f:
        np.save(f, label_array)
    with open(os.path.join(dataset_path, str(file_name),"info.txt"), "w") as f:
        f.write(f"Slice array: {slice_array.shape} Label array: {label_array.shape}")
        f.write(f'\nPlate number {plate_number}\nStrain: {strain}\nCrossdate: {crossdate}\nPrince position: {prince_position}\nNotice: It is bin {bin_number}. The width should be multiplied by {bin_number}.')

