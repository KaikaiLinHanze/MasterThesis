"""
File: video
Author: Kai
----------------------
It will be use to extract img data from certain region from experiment class.
"""
import os, shutil, zipfile, sys, json,cv2,datetime,imageio
sys.path.append("/home/ipausers/lin/Desktop/AMF/AMFtrack")
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment
from amftrack.pipeline.functions.image_processing.experiment_util import plot_full_image_with_features,plot_full,get_edge_from_node_labels
from amftrack.util.sys import update_plate_info,get_current_folders
from amftrack.pipeline.functions.image_processing.extract_width_fun import extract_section_profiles_for_edge

sys.path.append("/home/ipausers/lin/Desktop/Kai/Graduation-Project")
from models.config import *
import cv2
from datetime import datetime
from PIL import Image
import numpy as np

def main():
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
    
def extract_file(path, folder_name, file_name):
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
        
def image_for_video(exp, file_name:str,select_df,plate_number:str,region_size,save_img_path:str,downsizing=10):
    exp.load(select_df, suffix="")
    exp.load_tile_information(0)
    plot_full(exp,t=0,downsizing=downsizing,region = region_size, save_path= os.path.join(save_img_path,file_name+".png"))
    del exp
    plt.clf()
    # plot_full_image_with_features(exp,t=0,downsizing=10,region = region_size, save_path= os.path.join(save_img_path,file_name+".png"))
    return print(f'the mission for {file_name} is completed')

def img_to_video(img_files:list, save_img_path:str,save_file_name:str,choose_type = "normal",
                 width_files=None,save_width_path=None,width_value_list=None,
                 feature_files=None,save_feature_path=None):
    """
    img_files: list from os.listdir only file name
    save_img_path: str the main path to the folder you want to create movie.
    save_file_name: the name of the video. mp4
    choose_type: to handle the normal, width,and combine type of video
    
    width_files:list from os.listdir only file name
    save_width_path: str the main path to the folder you save width img
    width_value_list: list for the correct width value
    """
    img_path = [os.path.join(save_img_path,file) for file in img_files]
    date_list = ["".join(os.path.splitext(path)[0].split("/")[-1].split("_")[:2]) for path in img_path]
    datetime_list = [datetime.strptime(date, '%Y%m%d%H%M') for date in date_list]
    diff_list = [(date - datetime_list[0]) for date in datetime_list] 
    hr_list = [str(hr.seconds//3600+hr.days*24) + "hour(s)" for hr in diff_list] 
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

def extract_file(make_folder_path,move_file_path,unzip_file_path):
    os.mkdir(make_folder_path)
    shutil.move(move_file_path, make_folder_path)
    with zipfile.ZipFile(unzip_file_path, 'r') as zip_ref:
        zip_ref.extractall(make_folder_path)
    os.remove(unzip_file_path)


def get_edge_from_node(index,node_dict,node,exp):
    t = 0
    G, pos = exp.nx_graph[t], exp.positions[t]
    node_dict[index] = list(G.edges(node))[0]
    return node_dict

def get_slices_from_node(index,slice_dict,node_dict,exp,target_length=120):
    node1, node2 = node_dict[index]
    f_profiles = lambda edge: extract_section_profiles_for_edge(exp, 0, edge, resolution=5, offset=4, step=3,target_length=120)
    slices, coords1, coords2 = f_profiles(get_edge_from_node_labels(exp, 0, node1,node2))
    slice_dict[index] = slices
    return slice_dict

def get_width_from_slices(index,model,slice_dict,width_dict):
    predict_value = model.predict(slice_dict[index],verbose=0)
    width_dict[index] = predict_value
    return width_dict

def get_median_index_from_width(index,width_dict,median_index_dict):
    width = width_dict[index]
    median = np.median(width)
    if median > 1e5:
        median = np.median(np.delete(width, np.where(width == median)))
    median_index = np.where(width == median)[0]
    if len(median_index) == 0:
        median_index = (np.abs(width - median)).argmin()
    else:
        median_index = median_index[0]
    median_index_dict[index] = median_index
    width_dict[index] = median
    return width_dict, median_index_dict

def get_slice_from_median_index(index,median_index_dict,slice_dict,median_slice_dict):
    median_index = median_index_dict[index]
    slices = slice_dict[index]
    median_slice_dict[index] = slices[median_index]
    return median_index, median_slice_dict
    
def get_median_slice_width(exp,index,node,model,node_dict,slice_dict,width_dict,median_index_dict,median_slice_dict,target_length=120):
    get_edge_from_node(index,node_dict,node,exp)
    get_slices_from_node(index,slice_dict,node_dict,exp,target_length=120)
    get_width_from_slices(index,model,slice_dict,width_dict)
    get_median_index_from_width(index,width_dict,median_index_dict)
    get_slice_from_median_index(index,median_index_dict,slice_dict,median_slice_dict)
    return width_dict, median_slice_dict
if __name__ == '__main__':
    main()
