"""
File: data_preparation.py
Name: Kai
----------------------------------------
This file is used to load the dataset and data augmentation.
"""
import tensorflow as tf
import os
import keras
import imageio.v2 as imageio
from sklearn.model_selection import train_test_split
import numpy as np
from config import *


def data(select_path = "single_focus_0.0_train"):
    """
    This function is used to get the dataset you need from dataset folder.
    
    select_path: str, the name of choosed dataset
    
    return: array, features, array, label
    
    Notice:
    Make sure the shape is correct before training!
    """
    path = os.path.join(dataset_path,select_path)
    im_path = os.path.join(path, "slices.png")
    label_path = os.path.join(path, "labels.npy")
    im = imageio.imread(im_path)
    with open(label_path, "rb") as f:
        label = np.load(f)
    label = np.expand_dims(label,1)
    return im, label


def combine_dataset(data_list,value_size=1,test_size=0.2,random_state=42,shuffle=True):
    """
    combine several dataset together for training.
    
    data_list   : list, the folder name that you want to include them in the new dataset.
    value_size  : int, bin size of the data. For example, if the bin size is 2 you have to put 2 here as well.
                  Normally, the image size is 4096 × 3000. If it is resize to 2048 × 1500, you must put 2 instead of 1.
    test_size   : int, for split the training set and test set. Default is 0.2, the other o.8 for CV. 
                  If you do not use CV, you can change to 0.4 and split the test data again for validation. As the ratio of 0.6,0.2,0.2.
    random_state: int, pass an int for reproducible output 
    shuffle     : boolen, whether or not to shuffle the data before splitting.
    
    return      : array, feature for training
                  array, label for training
                  array, feature for testing
                  array, label for testing
    """
    X_train_list, y_train_list = [],[]
    X_test_list , y_test_list  = [],[]
    for dataset in data_list:
        X, y = data(dataset)
        y = y * value_size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,shuffle=shuffle)
        X_train_list.extend(X_train), y_train_list.extend(y_train)
        X_test_list.extend(X_test), y_test_list.extend(y_test)
    X_train_list, y_train_list, X_test_list , y_test_list = np.array(X_train_list), np.array(y_train_list), np.array(X_test_list) , np.array(y_test_list)
    return X_train_list, y_train_list, X_test_list , y_test_list


def data_preparation(X_test,size=OUTPUTSIZE):
    """
    This function is used for central crop the features.
    
    X_test: array, the feature that you want to central crop
    size  : int,   the size for cropping.
    
    return: array, features
    """
    if size == IMAGESIZE:
        return X_test
    
    else:
        data_preprocessing = keras.Sequential([
        tf.keras.Input(shape=(IMAGESIZE, 1)),
        center_crop_layer(IMAGESIZE, size)])
        return data_preprocessing(X_test)


def data_aug(X_train,size=OUTPUTSIZE):
    """
    This function is used for data augmentation.
    
    X_train: array, feature for changing the brightness, flip layer, 
    size   : int,   the size for output.
    
    return : array, features
    """
    if size == IMAGESIZE:
        data_preprocessing = keras.Sequential([
                tf.keras.Input(shape=(IMAGESIZE,1)),
                random_flip_layer(),])
        return data_preprocessing(X_train)
    else:
        data_preprocessing = keras.Sequential([
                tf.keras.Input(shape=(IMAGESIZE,1)),
                center_crop_layer(IMAGESIZE, size),
                random_flip_layer(),])
        return data_preprocessing(X_train)


@tf.function
def random_crop_slice(x, IMAGESIZE, OUTPUTSIZE, offset=0):
    """
    Randomly crop the -2 axis from original size to output size.
    
    x         : array, feature
    IMAGESIZE : int, the original size
    OUTPUTSIZE: int, the output size
    offset: used to reduce the range of random crop, offset = input_size/2 is equivalent to center_crop
    
    return : array, features
    
    Notice: original size must greater than output size + 2*offset
    """
    start_index = tf.experimental.numpy.random.randint(
        0 + offset,
        high=IMAGESIZE-OUTPUTSIZE - offset,
        dtype=tf.experimental.numpy.int64,)
    return x[..., start_index : start_index + OUTPUTSIZE, :]


def random_crop(IMAGESIZE, OUTPUTSIZE, offset=0):
    """
    This function is used for random cropping by using random_crop_slice function
    
    IMAGESIZE : int, the original size
    OUTPUTSIZE: int, the output size
    offset: used to reduce the range of random crop, offset = input_size/2 is equivalent to center_crop
    
    return : array, features
    """
    return tf.keras.layers.Lambda(
        lambda x: random_crop_slice(x, IMAGESIZE,OUTPUTSIZE, offset),
        name=f"random_crop_with_offset_{offset}",)


def random_flip_layer():
    """
    This function is used for random flip left to right
    https://www.tensorflow.org/api_docs/python/tf/image/random_flip_left_right
    
    return : array, features
    """
    return keras.layers.Lambda(lambda x: tf.image.flip_left_right(x),name="random_flip")


def random_brightness_layer(max_delta=0.4):
    """
    the layer for random brightness the img
    https://www.tensorflow.org/api_docs/python/tf/image/random_brightness
    
    max_delta: float, must be non-negative, randomly picked in the interval [-max_delta, max_delta]
    
    return : array, features
    """
    return keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta),name="random_brightness")

def center_crop(x,margin):
    """
    To remove the vector (left and right parts in the image)
    
    x     : array, the feature for cropping
    margin: int, the bounder that must be remove
    
    return : array, features
    """
    return x[..., margin:-margin, :]

def center_crop_layer(IMAGESIZE,OUTPUTSIZE):
    """
    The layer for remove the vertor (left and right parts in the image) by using the center crop function.
    
    IMAGESIZE : int, the original size
    OUTPUTSIZE: int, the output size    
    
    return    : array, the 
    """
    return keras.layers.Lambda(lambda x: center_crop(x,(IMAGESIZE-OUTPUTSIZE)//2),name="center_crop")

if __name__ =="__main__":
    X_train,y_train = data()
    print(data_preparation(X_train).shape)
