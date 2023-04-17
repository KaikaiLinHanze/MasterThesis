# MasterThesis
This is a repository for the graduation project for the Master thesis in Data Science for Life Sciences at Hanze University of Applied Sciences.

Arbuscular mycorrhizal fungi (AMF) are symbiotic fungi that form complex networks underground to exchange nutrients with carbon resources from their host plants. The structure of the networks will play an important role in trading dynamics. The fungus may decide to invest in a different structure responsible for absorbing phosphorous (branched absorbing structure (BAS)) or exploration (runner hyphae (RH)). The fungus will also invest in different widths of hyphae to optimize its resource acquisition strategy. In this thesis, the evolution of hyphal width was followed over time and space to unravel the strategy of the fungus in terms of width allocation. A convolutional neural network (CNN) model with a mean absolute error of 0.759 µm was developed to track the width evolution from a high-resolution automated imaging setup. The one-dimension vector from image data was extracted as a feature and the actual width was computed by pixel from high magnification image as a label for training the model. AMF can double its width in 40 hours and stay at a stable phase after 58 hours. The vasculature network formed by AMF does not follow Murray’s law. There is a threshold to distinguish between the RH to BAS in the junction.

Student: Kai-Kai Lin ka.lin@st.hanze.nl  
Supervisor: Tsjerk Wassenaar t.a.wassenaar@pl.hanze.nl  
Daily supervisor: Coretin Bisot C.Bisot@amolf.nl  

# Research questions  
•	How does the width evolve between different hyphae segments over time?  
•	Is there a relationship between the width of the different hyphae edges at the intersection?  
•	What criteria can be used to categorize RH and BAS?  

# Requirements

•	Python 3.10.6
•	Numpy  
•	MatplotLib  
•	Scikit-learn  
•	Tensorflow  
•	Keras  

# Setup
## 1. Clone the repository to your computer:
```
git clone https://github.com/KaikaiLinHanze/MasterThesis.git
```
## 2. Create a virtual environment.
```
virtualenv --python=python3 project
```
## 3. Launching environment:
```
source project/bin/activate
```
## 4. Install all of the requirements (make sure you already launched the environment)
```
pip3 install -r requirements.txt
```
## 5. Additional packages to install:
```
git clone https://github.com/Cocopyth/AMFtrack.git
git clone https://github.com/gattia/cycpd.git
cd cycpd
sudo python setup.py install
```
### Install Fiji:  
Chose a location on the local machine and download:  
https://imagej.net/software/fiji/downloads

### Install AniFilters:  
Chose a location on the local machine and download:  
http://forge.cbp.ens-lyon.fr/redmine/projects/anifilters

## 6. Create a Local.env file
Create a text file named local.env in the AMFTrack folder (for example: touch local.env)
```
DATA_PATH=/Users/kai/Downloads/graduation/final_git/MasterThesis/datasets
FIJI_PATH=/Users/kai/Downloads/graduation/AMFtrack/Fiji.app
TEMP_PATH=/Users/kai/Downloads/graduation/final_git/MasterThesis/tmp
STORAGE_PATH=/Users/kai/Downloads/graduation/final_git/MasterThesis/tmp
PASTIS_PATH=/home/cbisot/anis_filter/anifilters/bin/ani2D 
#the path to the executable of anisotropic filtering
SLURM_PATH=/scratch-shared/amftrack/slurm #this is for parallelizez job on snellius
SLURM_PATH_transfer=/data/temp
DROPBOX_PATH = /Users/kai/Downloads/graduation/

#For Dropbox transfers, ask dropbox admin for these values
APP_KEY=________
APP_SECRET=________
REFRESH_TOKEN =________
USER_ID=________
```

## 7. Path  
Before you run jupyter notebook make sure you already changed the path in src folder config file.

# Structure
```
     -----
    |     |
    |--- Experiment : the folder to save all of the experiment data
    |     |
    |     |-- the raw data for measure all of the hyphal width
    |     |
    |     |-- the data for examining unceratinity
    |     |
    |     |-- xlsx file for node, edge, position
    |     |
    |--- Plot data : the folder to save all of the data that was plotted
    |     |
    |     |-- Feature_outlier: the outlier feature
    |     |
    |     |-- GroundTruthML_Performance: the performance of the new model for 50X magnification. It will be use in the future.
    |     |
    |     |-- Labelme_uncertainity: the data for detecting manual annotation imprecision and width variation 
    |     |
    |     |-- Model_performance: the performance of the CNN model and also compare with other model
    |     |
    |     |-- Murray's law: the data for Murray's law and use different coefficient for seperate RH and BAS
    |     |
    |     |-- Width: Width growing for local view and global view
    |     |
    |--- datasets: the data set for training the CNN model ( with different focus, width, illumination
    |     |
    |--- model: with final model for the 2X and 50X magnificaion. In this research, only the 2X one was used.
    |     |
    |--- notebook: save the jupyter notebook for all of the analysis that I did during the intern
    |     |
    |--- src: the folder to save all of the scripts. Before run the code please change the path in config.
    |     |
    |     |-- config: set up the SIZE of the image and output for machine learning model
    |     |
    |     |-- data_prepare: for loading dataset and for data augmentation
    |     |
    |     |-- exp_surf: used to navigate the PRINCE data for experience. for node, edge data
    |     |
    |     |-- make_dataset: make dataset for training model. get segment, slice, width 
    |     |
    |     |-- model: for training, evaluating models. load model, tuning, bootstrapping
    |     |
    |     |-- plot: plot different output. model structure, performance, 
    |     |
    |     |-- video: extract img data from certain region from experiment class. for node, edge, slice, width data
    |     |
     -----
``` 
