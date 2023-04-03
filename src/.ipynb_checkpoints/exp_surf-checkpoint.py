"""
File: exp_surf.py
Name: Kai
----------------------------------------
This file is used to navigate the PRINCE data for experience.
"""
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
from config import *
sys.path.append(AMFTracker_path)
from amftrack.pipeline.functions.image_processing.experiment_util import get_edge_from_node_labels,get_all_nodes
from amftrack.pipeline.functions.image_processing.extract_width_fun import extract_section_profiles_for_edge,get_width_info_new
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Node,Experiment
from amftrack.pipeline.functions.post_processing.util import is_in_study_zone


def get_RH_BAS_edge(exp,select_node1,select_node2):
    """
    This function will get the edge by the shortest path.
    
    exp         : Experiment class
    select_node1: int
    select_node2: int
    
    return: list, with Node object
            list, with Edge object
            list, with Edge object
    Notice: 
    Remove the edge that does not make sense. 
    For example, it is RH but grouped in BAS.
    """
    t = 0
    G, pos = exp.nx_graph[t], exp.positions[t]
    nodes = []
    node_label=[]
    all_edge = []
    for path in sorted(nx.shortest_path(G, select_node1, select_node2)):
        node_label.append(path)
        nodes.append(Node(path,exp))
        for i in list(G.neighbors(path)):
            nodes.append(Node(i,exp))
            edge = get_edge_from_node_labels(exp, 0, path, i)
            all_edge.append(edge)
    RH_edge = []
    for index in range(len(node_label[:-1])):
        node1, node2 = node_label[index], node_label[index+1]
        edge = get_edge_from_node_labels(exp, 0, node1, node2)
        if edge is not None:
            RH_edge.append(edge)
    BAS_edge = list(set(all_edge).difference(RH_edge))
    return nodes, RH_edge,BAS_edge

def remove_BAS_edge(exp,RH_edge,BAS_edge,unwant_edge):
    """
    This function will remove the unwanted edge.
    
    exp         : Experiment class
    RH_edge     : list, with Edge object
    BAS_edge    : list, with Edge object
    unwant_edge : list, with tuple
    
    return: list, with Edge object
    
    For example: remove edge(32,45),and edge(78,89). [(32,45),(78,89)]
    """
    remove_edge = [get_edge_from_node_labels(exp, 0, node1, node2) for node1, node2 in unwant_edge]
    for i in remove_edge:
        for j in range(len(BAS_edge)-1):
            if i == BAS_edge[j]:
                del BAS_edge[j]
        if i == BAS_edge[-1]:
            del BAS_edge[-1]
    return BAS_edge

def compute_edge_ratio(exp,model,RH_edge,BAS_edge):
    """
    This function is used for computing the std of RH, std of BAS and ratio between RH and BAS.
    
    exp         : object, Experiment class
    model       : object, the model object for predict the width
    RH_edge     : list, with Edge object
    BAS_edge    : list, with Edge object
    
    return: list, with RH width value
            list, with BAS width value
    """
    f_profiles = lambda edge: extract_section_profiles_for_edge(
        exp, 0, edge, resolution=5, offset=5, step=1,target_length=120)
    RH_value = []
    BAS_value = []
    for edge in RH_edge:
        slices, coords1, coords2 = f_profiles(edge)
        RH_value.append(np.median(model.predict(slices,verbose=0)))
    for edge in BAS_edge:
        slices, coords1, coords2 = f_profiles(edge)
        BAS_value.append(np.median(model.predict(slices,verbose=0)))
    print("RH std:{} mean:{}".format(np.std(RH_value), np.mean(RH_value)))
    print('BAS std:{} mean:{}'.format(np.std(BAS_value), np.mean(BAS_value)))
    print('BAS/RH : {}'.format(np.mean(BAS_value)/np.mean(RH_value)))
    return RH_value,BAS_value

def get_width_from_exp(exp,t=0):
    """
    This function is used for extracting the width from exp object.
    
    exp : object, Experiment class
    t   : int, time step
    
    return: list, width
    """
    width_dict = get_width_info_new(exp, t, resolution=50)
    collect = []
    for _, width in width_dict.items():
        if width != 0:
            collect.append(width)
    return collect

def get_width_hist(collect,save_width_path,file,xmax=18,ymax=500,save=True):
    """
    This function is used to plot the width histogram.
    
    collect        : list, width
    save_width_path: str, path of folder for saving the width histogram
    file           : str, file name
    xmax           : int, the x lim of the plot.
    ymax           : int, the y lim of the plot.
    save           : boolen, for choosing save or not.
    """
    plt.hist(collect,color='blue')
    plt.xlabel('Width ($\mathit{\mu m}$)')
    plt.ylabel("Frequency")
    plt.ylim(0,ymax)
    plt.xlim(0,xmax)
    if save:
        plt.savefig(fname= save_width_path+file+".png")
        plt.clf()
    else:
        plt.show()

def get_width_hist_combine(exp,t,save_width_path,file,save=True):
    """
    This function is to combine extracting and plotting width together.
    
    exp            : object, Experiment class
    t              : int, time step
    save_width_path: str, path of folder for saving the width histogram
    file           : str, file name
    save           : boolen, for choosing save or not.
    
    return: str
    """
    collect = get_width_from_exp(exp,t=t)
    get_width_hist(collect,save_width_path,file,save=save)
    return f'finish {file}'

def get_edge_width_from_analysis(analysis_path):
    """
    This function is used to get the information for whole plate after analyzing by supercomputer.
    
    analysis_path: str, path for analysis folder
    
    return: object, df
    """
    path_time_edge = os.path.join(analysis_path, "time_edge_info")
    path_save = os.path.join(analysis_path, "folder_info.json")
    folders_plate = pd.read_json(path_save)
    folders_plate = folders_plate.reset_index()
    folders_plate = folders_plate.sort_values("datetime")
    json_paths = os.listdir(path_time_edge)
    tables = []
    for path in json_paths:
        try:
            index = int(path.split("_")[-1].split(".")[0])
            line = folders_plate.iloc[index]
            table = pd.read_json(os.path.join(path_time_edge, path))
        except:
            print(os.path.join(path_time_edge, path))
            continue
        table = table.transpose()
        table = table.fillna(-1)
        table["time_since_begin_h"] = (line["datetime"] - folders_plate["datetime"].iloc[0])
        table["folder"] = line["folder"]
        table["datetime"] = line["datetime"]
        tables.append(table)
    time_edge_info_plate = pd.concat(tables, axis=0, ignore_index=True)
    time_edge_info_plate.reset_index(inplace=True, drop=True)
    return time_edge_info_plate
    

def load_study_zone(exp,path_code = AMFTracker_path ):
    """
    This function is used to set the study zone for exp.
    
    exp      : object, Experiment class
    path_code: str, AMFTracker_path
    """
    loc_load = os.path.join(path_code, "pipeline", "functions", "post_processing", "default_param")
    exp.center = np.load(os.path.join(loc_load, "center.npy"))
    exp.orthog = np.load(os.path.join(loc_load, "orthog.npy"))
    exp.reach_out = np.load(os.path.join(loc_load, "reach_out.npy"))
    exp.num_trunk = np.load(os.path.join(loc_load, "num_trunk.npy"))

def get_edge_node_list_by_degree(exp,t=0,degree=3,radius=1000,dist=150):
    """
    This function is used to select the edges by degree.
    exp     : object, Experiment class
    t       : int, time step
    degree  : int, the degree of the edge
    radius  : int, the radius of the plate.
    dist    : int, the distance from the bottom of the PRINCE image.
    
    return: list, Edge
            list, Node
    
    Notice: radius and dist already have default value. 
            You can change the value if you know how it shift.
            For example: 1. if you find that the edge is below the compartment, you can increase the dist
                         2. if you find that the edge is out of the plate boundry, you can decrease the radius
    """
    load_study_zone(exp)
    node_list = get_all_nodes(exp, t)
    G, pos = exp.nx_graph[t], exp.positions[t]
    collect = []
    for i in node_list:
        if i.degree(t) == degree:
            if sum(is_in_study_zone(i,t,radius,dist)) == 2:
                collect.append(i)
    return [G.edges(int(str(i))) for i in collect], collect

def get_widths_edges_by_degree(exp,t,edge_list,width_threshold=0,degree = 3):
    """
    This function is used to get the width from edge list it can also set width threshold to remove the bias.
    
    exp            : object, Experiment class
    t              : int, time step
    width_threshold: int, remove the edge that is below the threshold
    degree         : int, the degree of the edge
    
    return: list, width
            list, edge
    """
    width_result = []
    edge_result = []
    for edges in edge_list:
        collect_width = []
        collect_edge = []
        for node1,node2 in edges:
            edge = get_edge_from_node_labels(exp,t,node1,node2)
            if edge.width(0) > width_threshold:
                collect_width.append(edge.width(0))
                collect_edge.append(edge)
        if len(collect_width) == degree:
            collect_width.sort()
            width_result.append(collect_width)
            edge_result.append(collect_edge)
    return width_result, edge_result

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