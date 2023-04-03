"""
File: config.py
Name: Kai
----------------------------------------
This file is used to set up the SIZE of the image and output for machine learning model.
Commonly used paths are also included here.
You can change the download path, AMFTracker_path to run the code.
"""

IMAGESIZE = 120
OUTPUTSIZE = 80

download_path = "/Users/kai/Downloads/graduation/final_git/MasterThesis/"
AMFTracker_path = "/Users/kai/Downloads/graduation/AMFtrack"

dropbox      = "/run/user/357100579/gvfs/smb-share:server=sun.amolf.nl,share=shimizu-data/home-folder/Kaikai"
labelme_path = download_path + "Experiment/"
dataset_path = download_path + "datasets/"
model_path   = download_path + "model/"
src_path     = download_path + "src/"
storage_path = download_path + "tmp/"
