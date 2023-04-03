"""
File: plot.py
Name: Kai
----------------------------------------
This file is used to plot different output.
"""
import matplotlib.pyplot as plt
from tensorflow import keras
from config import *
from data_prepare import *
import numpy as np
from model import * 

def evaluate_epoch(history,type="loss"):
    """
    for plotting different histories of hp.
    
    history: object, for putting the model history during fitting.
    type   : str,    for choose the type of output.
    
    return : str
    """
    import matplotlib.pyplot as plt
    if type == "loss":
        plt.plot(history.history["loss"],label="Loss")
        plt.plot(history.history["val_loss"],label="Val_Loss")
        plt.ylabel('Loss')
        plt.ylim(0, 20)
    if type == "mae":
        plt.plot(history.history["mean_absolute_error"],label="MAE")
        plt.plot(history.history["val_mean_absolute_error"],label="Val_MAE")
        plt.ylabel('MAE')

    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    return print(f'{type} epoch plot execute!')


def model_structure(model,output_name):
    """
    a function for plot the model structure
    
    model       : object
    output_name : str, the path for saving the structure.
    
    
    Notice      : the ouput for the result is from 
                  https://netron.app 
    """
    keras.utils.plot_model(model,output_name)
    return f'Your model structure already save as {output_name}'


def plot_seperate_dataset(model,dataset:str,value_size:int,crop_value:int):
    """
    to plot the dataset seperately
    
    model     : object, the model you want to check the performance
    dataset   : str, the dataset you want to check
    value_size: int, if you set the bin to 2, must set it as 2
    crop_value: int, for central or random cropping
    
    return    : object, plt for plotting.
    
    Notice    : x-axis is actual, y-axis is prediction
    """
    X, y = data(dataset)
    y = y * value_size
    X_train_iter, X_test_iter, y_train_iter, y_test_iter = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_test = data_preparation(X_test_iter,crop_value)
    plt.plot(model.predict(X_test), y_test_iter,"o",label = dataset)
    return plt

def get_regression_line(X_test, y_test, x_range=[0, 14]):
    """
    for plotting the X,y with regression line and the R square.
    
    X_test: int, for plotting the 
    y_test: int, 
    """
    X = np.array([1,2,3,4]).reshape(4,1)
    y = np.array([1,2,3,4]).reshape(4,1)
    lr = LinearRegression().fit(X,y)
    r_sq = lr.score(X_test, y_test)
    print(f"coefficient of determination: {r_sq}")
    y_pred = lr.intercept_ + lr.coef_ * x_range
    return x_range, y_pred[0]


def compare_predict_extact(model,X_test,y_test,x_range=[0, 14]):
    """
    model
    """
    plt.plot(y_test,model.predict(X_test), "o", label="width")
    x, y = get_regression_line(model.predict(X_test), y_test, x_range)
    plt.plot(x, y, label="actual line")
    plt.ylabel('Predict width')
    plt.xlabel('Actural width')
    plt.legend()
    plt.title("Check the accuracy of estimation")
    plt.show()


def compare_different_model(model, X_test, y_test, set_title="HB", total=3, plot_number=1,x_range=[0,14]):
    """
    
    """
    x, y = get_regression_line(model.predict(X_test), y_test,x_range)
    sub = plt.subplot(1, total, plot_number)
    sub.plot(x, y)
    sub.plot(model.predict(X_test), y_test, "o", label="width")
    sub.set(title=set_title, xlabel="predict", ylabel="actual")
    return sub


def hp_model_output_information(model,X_train,y_train,X_test,y_test):
    """
    This function is get the output from hp.
    
    model              : object, for evaluate the model
    X                  : array, for feature for the model
    y                  : array, for label for the model
    X_test             : array, for feature for the model
    y_test             : array, for label for the model
    
    return             : model, object of trained model
                         list, objects of history for recording of training loss values and metrics values.
                         list, objects of the evaluate result.
    """
    best_model, best_model_hp, history= evaluate_hp_model(model,X_train,y_train,X_test,y_test)
    evaluate_epoch(history,type="loss")
    evaluate_epoch(history,type="mae")
    compare_predict_extact(best_model,X_test,y_test)
    return best_model, best_model_hp, history