"""
File: model.py
Name: Kai
----------------------------------------
This file is used for training, evaluating models.
"""

import tensorflow as tf
import sys, imageio, os
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
import numpy as np
import keras_tuner as kt
from sklearn.utils import resample
from config import *
from data_prepare import *


def first_model():
    """
    The first structure for the CNN model.
    Did not use hp.
    """
    input = keras.layers.InputLayer(input_shape=(OUTPUTSIZE,1)) # image shape is 120 dimension
    scaling = keras.layers.Rescaling(1 / 255)
    Conv1 = keras.layers.Conv1D(
        filters = 300, kernel_size=12,
        strides = 3,   activation="relu",
        name    ="conv1", activity_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-5))
    pool1 = keras.layers.MaxPooling1D(2)
    Conv2 = keras.layers.Conv1D(
        filters = 320, kernel_size=8,
        strides =3,     activation="relu",
        name="conv2",  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))
    pool2 = keras.layers.MaxPooling1D(2)
    flatten = keras.layers.Flatten()
    Dense1 = keras.layers.Dense(units = 340,activation="relu",name="dense1",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))
    Dense2 = keras.layers.Dense(units = 220,activation="relu",name="dense2",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))
    output = keras.layers.Dense(units = 1,activation=None,name="output",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))
    model = keras.models.Sequential([input,scaling,Conv1,pool1,Conv2,pool2,flatten,Dense1,Dense2,output])
    return model


def evaluate_model(model, X, y, X_test,y_test,learning_rate_value=0.001,crop_size=100,n_split=3,shuffle=True,random_state=42,batch_size=5,epochs=60):
    """
    This function is used for evaluate the model.
    
    model              : object, for evaluate the model
    X                  : array, for feature for the model
    y                  : array, for label for the model
    X_test             : array, for feature for the model
    y_test             : array, for label for the model
    learning_rate_value: int, for setting the learning rate
    crop_size          : int, for setting the crop size ( for central crop )
    n_split            : int, for cross validation
    shuffle            : boolen, whether or not to shuffle the data before splitting.
    random_state       : int, pass an int for reproducible output 
    
    return             : model, object of trained model
                         list, objects of history for recording of training loss values and metrics values.
                         list, objects of the evaluate result.
    """
    evaluations = []
    historys = []
    kf = KFold(n_splits=n_split,shuffle=shuffle,random_state=random_state)
    for (train_index, val_index) in kf.split(X):
        train_feature = X[train_index]
        train_label   = y[train_index]
        val_feature  = X[val_index]
        val_label    = y[val_index]
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate_value),
            loss="mean_squared_error",
            metrics="mean_absolute_error")
        history = model.fit(
                        x=data_preparation(train_feature,crop_size),
                        y=train_label,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(data_preparation(val_feature,crop_size), val_label),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_mean_absolute_error",min_delta=1e-3,patience=4)])
        evaluate = model.evaluate(data_preparation(X_test,crop_size), y_test)
        evaluations.append(evaluate)
        historys.append(history)
    return model,historys,evaluations


def build_model_CNN(hp):
    """
    The architecture of the CNN model for hyper-tuning.
    
    hp : to define the hyperparameters during model creation.
    
    return: model with best hp 
    """
    input_size = hp.Fixed("input", OUTPUTSIZE)
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(input_size, 1)))
    model.add(keras.layers.Rescaling(1 / 127.5, offset=-1))
    for i in range(hp.Int("num_conv_layers", 1, 4, default=2)):
        model.add(keras.layers.Conv1D(
            filters=hp.Int(f"filters_{i + 1}", min_value=32, max_value=256, step=32),
            kernel_size=hp.Choice(f'kernel_size_{i}', values=[10,11,12,13,14,15,16, 17,18,19,20]),
            activation="relu",
            name=f"conv_{i + 1}",
            kernel_regularizer=keras.regularizers.L1(
                hp.Float("kernel_regulizer", min_value=1e-5, max_value=1e-1, sampling="log")),
            activity_regularizer=keras.regularizers.L1(
                hp.Float("activity_regulizer", min_value=1e-5, max_value=1e-1, sampling="log")),
            bias_regularizer=keras.regularizers.L1(
                hp.Float("activity_regulizer", min_value=1e-5, max_value=1e-1, sampling="log"))))
    hp_pooling = hp.Choice(f'pooling_{i}', values=["MP", "AP", "No pool"])
    hp_padding = hp.Choice('padding_' + str(i), values=['valid', 'same'])
    if hp_pooling == "MP":
        model.add(keras.layers.MaxPooling1D(hp.Int(f"MP", min_value=1, max_value=4, step=1),padding = hp_padding))
    if hp_pooling == "AP":
        model.add(keras.layers.AveragePooling1D(hp.Int(f"AP", min_value=1, max_value=4, step=1),padding = hp_padding))
    if hp.Boolean("batch_normalization", default=False):
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    for j in range(hp.Int("num_dense_layers", 1, 4, default=2)):
        model.add(keras.layers.Dense(
            units=hp.Int("dense_size", min_value=8, max_value=256, step=32), activation="relu",
            name=f"dense_{j + 1}",
            kernel_regularizer=keras.regularizers.L1(
                hp.Float("kernel_regulizer", min_value=1e-5, max_value=1e-1, sampling="log")),
            activity_regularizer=keras.regularizers.L1(
                hp.Float("activity_regulizer", min_value=1e-5, max_value=1e-1, sampling="log")),
            bias_regularizer=keras.regularizers.L1(
                hp.Float("activity_regulizer", min_value=1e-5, max_value=1e-1, sampling="log"))))
        model.add(keras.layers.Dropout(hp.Float(f"dropout_{j + 1}", 0, 0.5, step=0.1, default=0.2)))
    model.add(keras.layers.Dense(units=1, activation=None, name="output"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", min_value=1e-5, max_value=1e-1, sampling="log", default=1e-3)),
        loss="mean_squared_error",
        metrics="mean_absolute_error")
    return model


def evaluate_hp_model(model,X_train,y_train,X_test,y_test,epochs=120,batch_size=5):
    """
    for evaluate the hp model
    
    model     : object, put train model here
    X_train   : array, for feature training data
    y_train   : array, for label training data
    X_test    : array, for feature test data
    y_test    : array, for label test data
    epochs    : int, for setting the epochs
    batch_size: int for setting the batch size
    
    return    : object, model
                object, the best hyperparameters
                object, history for recording of training loss values and metrics values.
    """
    best_model = model.get_best_models(1)[0]
    best_model_hp = model.get_best_hyperparameters(1)[0]
    model_output = model.hypermodel.build(best_model_hp)
    history = model_output.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_test,y_test))
    print(best_model.evaluate(X_test,y_test))
    print(best_model_hp.values)
    return best_model, best_model_hp, history


def load_model(model_save_path = model_path, select_model = "default_CNN_model.h5"):
    """
    For load the model and plot the summary.
    
    model_save_path : str, the path to where model is saved
    select_model    : str, the model you want to get
    
    return          : object model
    """
    model_select = os.path.join(model_save_path,select_model)
    model = keras.models.load_model(model_select)
    model.summary()
    return model


def load_model_list(model_list,model_save_path= model_path):
    """
    for loading several model at the same time.
    
    model_save_path: str, the path to where model is saved
    model_list     : list, the model file name that you want to get
    
    return         : list, model
                     list, learning rate
    """
    from keras import backend
    output, learning_rate=[], []
    for model_name in model_list:
        model = load_model(model_save_path,model_name)
        output.append(model)
        learning_rate.append(backend.get_value(model.optimizer.lr))
    return output, learning_rate



def tuning_model(model, callbacks, X_train, y_train, X_test, y_test, model_type="BO", project="CNN_model"):
    """
    This function is used for tuning different model (include BayesianOptimization, and Hyperband )
    
    model     : object, keras function (with hp), for the model you want to train for this tuning model
    callbacks : object, setup a call back that you want to put in tuning
    X_train   : array, as the feature. Do not forget to make sure the size. (in model and original data)
    y_train   : array, as the label
    X_test    : array, as the feature. Do not forget to make sure the size. (in model and original data)
    y_test    : array, as the label
    model_type: str, to choose different type. BO for BayesianOptimization, HB for Hyperband
    project   : str, for setting the project name. You must rename it everytime.
    
    return    : object, tuner 
    """
    if model_type == "BO":
        tuner = kt.BayesianOptimization(
                model,
                objective="val_mean_absolute_error",
                max_trials=100,
                num_initial_points=50,
                alpha=0.0001,
                beta=2.6,
                seed=11,
                directory='.',
                project_name= project,
                hyperparameters=None,
                tune_new_entries=True,
                allow_new_entries=True,)
    if model_type == "HB":
        tuner = kt.Hyperband(
            model,
            objective="val_mean_absolute_error",
            project_name=project,
            seed=11,
            overwrite=True,
            hyperparameters=None,
            tune_new_entries=True,
            allow_new_entries=True,)
        
    tuner.search(
        X_train,
        y_train,
        validation_data = (X_test,y_test),
        epochs=100,
        callbacks= callbacks,)
    return tuner

def statistic_evaluate(model_list, X_test,y_test,crop_list,statistic_type="MAE",n_iterations=1000,n_samples=50):
    """
    This function use bootstrapping to resample the X test, and y test to calculate the MAE, RE, ME.
    
    model_list    : list, trained model list
    X_test        : array, the feature from the dataset
    y_test        : array, the label from the dataset
    crop_list     : list, with different value to fit the crop size of the model
    statistic_type: str, with different type. MAE for evaluate the MAE
                                              RE for evaluate the RE
                                              ME for evaluate the ME
    n_iterations  : int, insert the value that you want to resample
    n_samples     : int, the sample you will get from the each resample times.
    
    return        : array, the accuracy of after certain time
    """
    acc_list = []
    counter = 0
    for index in range(len(model_list)):
        acc_list.append([])
    for _ in range(n_iterations):
        X_bs, y_bs = resample(X_test, y_test, replace=True,n_samples=n_samples)
        if 0 in y_bs:
            counter += 1
        for index in range(len(model_list)):
            crop_size = crop_list[index]
            X_bs_input = data_preparation(X_bs,crop_size)
            if statistic_type=="MAE":
                loss, acc = model_list[index].evaluate(X_bs_input, y_bs, verbose=0)
                acc_list[index].append(acc)
            if statistic_type=="RE":
                if 0 not in y_bs:
                    acc = np.mean(abs(model_list[index].predict(X_bs_input,verbose=0)-y_bs)/y_bs)
                    acc_list[index].append(acc)
            if statistic_type=="ME":
                acc = np.mean(model_list[index].predict(X_bs_input,verbose=0)-y_bs)
                acc_list[index].append(acc)
    if statistic_type=="RE":
        print(f"Get {counter} time(s) 0 in label, so it will not include in the output.")
    else:
        print(f"Get {counter} time(s) 0 in label.")
    return np.array(acc_list)




if __name__ =="__main__":
    X_train, y_train = data("single_focus_0.0_train")
    X_test, y_test = data("single_focus_0.0_test")
    X_train = data_aug(X_train)
    X_test = data_preparation(X_test)
    callbacks = [
        keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            # "no longer improving" being defined as "no better than 1e-3 less"
            # "no longer improving" being further defined as "for at least 3 epochs"
            monitor="val_loss",
            min_delta=1e-3,
            patience=3,
            verbose=1)]
    tuner_baye = kt.BayesianOptimization(
        build_model_CNN,
        objective="val_mean_absolute_error",
        max_trials=100,
        num_initial_points=50,
        alpha=0.0001,
        beta=2.6,
        seed=11,
        directory='.',
        project_name='CNN_nice_focus_model',
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True, )
    tuner_baye.search(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        callbacks=callbacks, )


