"""
File: video
Author: Kai
----------------------
It is used to extract img data from certain region from experiment class.
"""
import os, shutil, zipfile, sys, json,cv2,imageio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import numpy as np
from config import *
sys.path.append(AMFTracker_path)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment
from amftrack.pipeline.functions.image_processing.experiment_util import plot_full_image_with_features,plot_full,get_edge_from_node_labels
from amftrack.util.sys import update_plate_info,get_current_folders
from amftrack.pipeline.functions.image_processing.extract_width_fun import extract_section_profiles_for_edge

        
def image_for_video(directory, select_df,region_size,file_name:str,save_img_path=storage_path,downsizing=10):
    """
    create frame for the video
    
    directory    : str, the path to the plate for example: "521_20230104"
    select_df    : object, folder_df = get_current_folders(directory)
                         file_names = folder_df.folder.tolist()
                         select_df = folder_df[(folder_df["folder"] == file_names[i])]  # i ,int for the file you want to load in.
    
    region_sizef : list, the format will be similar like this :[[20500, 23800], [20800,24300]]
    file_name    : str, for save the file as your insert name
    save_img_path: str, the path to saving folder save img path
    
    return       : str
    """
    exp = Experiment(directory)
    exp.load(select_df)
    exp.load_tile_information(0)
    plot_full(exp,t=0,downsizing=downsizing,region = region_size, save_path= os.path.join(save_img_path,file_name+".png"))
    del exp
    plt.clf()
    # plot_full_image_with_features(exp,t=0,downsizing=10,region = region_size, save_path= os.path.join(save_img_path,file_name+".png"))
    return print(f'the mission for {file_name} is completed')


def get_hr_list_from_folder(folder_list):
    """
    This function is used to extract the file name and change it to the datetime.
    
    folder_list: list, with the name in 20220504_1900_Plate02 format.
    
    return     : list
    """
    date_list = ["".join(os.path.splitext(path)[0].split("/")[-1].split("_")[:2]) for path in folder_list]
    datetime_list = [datetime.strptime(date, '%Y%m%d%H%M') for date in date_list]
    diff_list = [(date - datetime_list[0]) for date in datetime_list]
    hr_list = [str(hr.seconds//3600+hr.days*24) + "hour(s)" for hr in diff_list] 
    return hr_list


def img_to_video(img_files:list, save_img_path:str,save_file_name:str,choose_type = "normal",
                 width_files=None,save_width_path=None,width_value_list=None,
                 feature_files=None,save_feature_path=None):
    """
    img_files        : list, from os.listdir only file name
    save_img_path    : str, the main path to the folder you want to create video.
    save_file_name   : str, the name of the video. mp4
    choose_type      : str, to handle the normal, width,and combine type of video
    
    width_files      : list, from os.listdir only file name
    save_width_path  : str, the main path to the folder you save width img
    width_value_list : list, for the correct width value
    
    feature_files    : list, from os.listdir only file name
    save_feature_path: str, the main path to the folder you save feature img
    
    return           : list
    """
    img_path = [os.path.join(save_img_path,file) for file in img_files]
    hr_list = get_hr_list_from_folder(img_path)
    imgs = [cv2.imread(path) for path in img_path]
    for index,img in enumerate(imgs):
        pos = img.shape[0]//10, img.shape[1]//10
        cv2.putText(img=img, text=hr_list[index], org=pos, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 0, 0),thickness=3)
        if choose_type == "width":
            width_path = [os.path.join(save_width_path,file) for file in width_files]
            width = cv2.imread(width_path[index]) 
            img[img.shape[0]-width.shape[0]:img.shape[0], img.shape[1]-width.shape[1]:img.shape[1]] = width
            pos2 = img.shape[1]-width.shape[1],img.shape[0]-width.shape[0]
            cv2.putText(img=img, text=str(np.around(width_value_list[index],1))+'um', org= pos2, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 0, 0),thickness=2)
        if choose_type == "all":
            width_path = [os.path.join(save_width_path,file) for file in width_files]
            width = cv2.imread(width_path[index]) 
            img[img.shape[0]-width.shape[0]:img.shape[0], img.shape[1]-width.shape[1]:img.shape[1]] = width
            pos2 = img.shape[1]-width.shape[1],img.shape[0]-width.shape[0]
            cv2.putText(img=img, text=str(np.around(width_value_list[index],1))+'um', org= pos2, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 0, 0),thickness=2)
            feature_path = [os.path.join(save_feature_path,file) for file in feature_files]
            feature = cv2.imread(feature_path[index])
            resized = cv2.resize(feature, (img.shape[1]//3, img.shape[0]//3), interpolation = cv2.INTER_AREA)
            img[img.shape[0]-resized.shape[0]:img.shape[0], :resized.shape[1]] = resized
    imageio.mimsave(os.path.join(save_img_path,save_file_name),imgs)
    return imgs


def get_extract_file_path(path, folder_name, file_name):
    """
    to get the path you want to extract the file
    
    path         : str, to the folder with all of the data with different time step. 
                        for example: "521_20230104" the code will go through the folder in this folder like 20230109_0149_Plate02
    folder_name  : str, the new folder name for collect the data you unzip for example :"Analysis"     or "Img"
    file_name    : str, the file name that you want to unzip.              for example :"Analysis.zip" or "Img.zip"
    
    return       : list, the path to make folder for saving the file
                   list, the path for where the file is.
                   list, the path to where you want to unzip the file
    """
    folder = os.listdir(path)
    folder.sort()
    make_folder_path_list,move_file_path_list,unzip_file_path_list = [],[],[]
    for i in folder:
        folder_path = os.path.join(path,i)
        make_folder_path = os.path.join(folder_path, folder_name)
        move_file_path = os.path.join(folder_path, file_name)
        unzip_file_path = os.path.join(folder_path,folder_name,file_name)
        make_folder_path_list.append(make_folder_path)
        move_file_path_list.append(move_file_path)
        unzip_file_path_list.append(unzip_file_path)
    return make_folder_path_list,move_file_path_list,unzip_file_path_list


def extract_file_mul(make_folder_path,move_file_path,unzip_file_path):
    """
    multiprocess version. to extract the file, remove the zip file. 
    
    make_folder_path: str, the path to make folder for saving the file
    move_file_path  : str, the path for where the file is.
    unzip_file_path : str, the path to where you want to unzip the file 
    """
    os.mkdir(make_folder_path)
    shutil.move(move_file_path, make_folder_path)
    with zipfile.ZipFile(unzip_file_path, 'r') as zip_ref:
        zip_ref.extractall(make_folder_path)
    os.remove(unzip_file_path)

    
def extract_file(path, folder_name, file_name):
    """
    for extract the file when you download the data from dropbox
    It will automatically unzip and remove the original file. 
    
    path         : str, to the folder with all of the data with different time step. 
                        for example: "521_20230104" the code will go through the folder in this folder like 20230109_0149_Plate02
    folder_name  : str, the new folder name for collect the data you unzip for example :"Analysis"     or "Img"
    file_name    : str, the file name that you want to unzip.              for example :"Analysis.zip" or "Img.zip"
    """
    folder = os.listdir(path)
    folder.sort()
    for i in folder:
        folder_path = os.path.join(path,i)
        make_folder_path = os.path.join(folder_path, folder_name)
        move_file_path = os.path.join(folder_path, file_name)
        unzip_file_path = os.path.join(folder_path,folder_name,file_name)
        os.mkdir(make_folder_path)
        shutil.move(move_file_path, make_folder_path)
        with zipfile.ZipFile(unzip_file_path, 'r') as zip_ref:
            zip_ref.extractall(make_folder_path)
        os.remove(unzip_file_path)

def get_edge_from_node(index,node_dict,node,exp):
    """
    Use to get the edge from given node. Only return the first edge.
    You must check whether the edge is the correct edge that you want.
    
    index    : int, to collect the file from different timestep
    node_dict: dict, the dict for collect the two node of the edge
    node     : int, the position that you want to detect
    exp      : object, Experiment
    
    return   : dict, the dict with index as key and edge as value.
    """
    t = 0
    G, pos = exp.nx_graph[t], exp.positions[t]
    node_dict[index] = list(G.edges(node))[0]
    return node_dict

def get_slices_from_node(index,slice_dict,node_dict,exp,target_length=120):
    """
    Use to get the slice from given node.
    You must check whether the edge is the correct edge that you want.
    
    index        : int, to save slice from slice_dict and extract the node from node_dict
    slice_dict   : dict, the dict for collect the slices.
    node_dict    : dict, the dict already collect the node
    exp          : object, Experiment
    target_length: int, to change the value that you want to extract the feature.
    
    return       : dict, the dict with index as key and slice as value.
    """
    node1, node2 = node_dict[index]
    f_profiles = lambda edge: extract_section_profiles_for_edge(exp, 0, edge, resolution=5, offset=4, step=3,target_length=target_length)
    slices, coords1, coords2 = f_profiles(get_edge_from_node_labels(exp, 0, node1,node2))
    slice_dict[index] = slices
    return slice_dict

def get_width_from_slices(index,model,slice_dict,width_dict):
    """
    Use to predict the width from you given slice
    
    index        : int, to width slice from width_dict and extract the slice from slice_dict
    model        : object, the trained tensorflow model that you would like to predict the width
    slice_dict   : dict, the dict for extract the slice from value
    width_dict   : dict, the dict for collect the width
    exp          : object, Experiment
    target_length: int, to change the value that you want to extract the feature.
    
    return       : dict, the dict with index as key and width as value.
    """
    predict_value = model.predict(slice_dict[index],verbose=0)
    width_dict[index] = predict_value
    return width_dict

def get_median_index_from_width(index,width_dict,median_index_dict):
    """
    Use to extract the index of the median width from width
    
    index             : int, to select the width from width dict and save into the median index dict
    width_dict        : dict, the dict for extract the width from different time point
    median_index_dict : dict, the dict for collect the median index
    
    return            : dict, the dict with index as key and median index of the width as value.
    """
    width = width_dict[index]
    median = np.median(width)
    # if median > 1e5:
    #     median = np.median(np.delete(width, np.where(width == median)))
    median_index = np.where(width == median)[0]
    if len(median_index) == 0:
        median_index = (np.abs(width - median)).argmin()
    else:
        median_index = median_index[0]
    median_index_dict[index] = median_index
    width_dict[index] = median
    return width_dict, median_index_dict

def get_slice_from_median_index(index,median_index_dict,slice_dict,median_slice_dict):
    """
    index             : int, to select the width from width dict and save into the median index dict
    width_dict        : dict, the dict for extract the width from different time point
    median_index_dict : dict, the dict for collect the median index
    
    return            : dict, the dict with index as key and median index of the width as value.
    """
    median_index = median_index_dict[index]
    slices = slice_dict[index]
    median_slice_dict[index] = slices[median_index]
    return median_index, median_slice_dict
    
def get_median_slice_width(exp,index,node,model,node_dict,slice_dict,width_dict,median_index_dict,median_slice_dict,target_length=120):
    """
    exp               : object, Experiment
    index             : int, to select different key for saving value
    node              : int, the position that you want to detect
    model             : object, the trained tensorflow model that you would like to predict the width
    node_dict         : dict, for collect and extract the node data
    slice_dict        : dict, for collect and extract the slice data
    width_dict        : dict, for collect and extract the width data
    median_index_dict : dict, for collect and extract the median index data
    median_slice_dict : dict, for collect and extract the slice of median width data
    target_length     : int, to change the value that you want to extract the feature.
    
    return            : dict, the width from different file
                        dict, the median slice from different file
    """
    get_edge_from_node(index,node_dict,node,exp)
    get_slices_from_node(index,slice_dict,node_dict,exp,target_length=target_length)
    get_width_from_slices(index,model,slice_dict,width_dict)
    get_median_index_from_width(index,width_dict,median_index_dict)
    get_slice_from_median_index(index,median_index_dict,slice_dict,median_slice_dict)
    return width_dict, median_slice_dict

if __name__ == '__main__':
    directory_groundtruths = os.path.join(storage_path,"video","521_20230104")
    directory = directory_groundtruths+"/"
    save_img_path = directory+"video_img/"
    update_plate_info(directory)
    folder_df = get_current_folders(directory)
    file_names = folder_df.folder.tolist()
    file_names.sort()
    plate_number = "521"
    number = len(os.listdir(os.path.join(storage_path,"video","521_20230104",'video_img')))-1
    for file_name in file_names[number:]:
        select = folder_df[(folder_df["Plate"] == plate_number) & (folder_df["folder"] == file_name)]
        image_for_video(directory, file_name,select,"521",[[16600, 19600], [17200,20400]], save_img_path,downsizing=1)
        del select
