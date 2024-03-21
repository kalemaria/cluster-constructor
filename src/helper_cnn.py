#!/usr/bin/python -tt

import zipfile
import os
import glob
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics
import seaborn as sns

from sklearn.utils import shuffle
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
	
    
    
def plot_class_balances(df, col):
    """plots the counts of classes at specific hierarchy level"""

    ser_counts = df[col].value_counts()
    ser_counts.plot.bar()
    plt.title(col + ' Counts \n(classes={})'.format(ser_counts.shape[0]))
    
    plt.show()
    
    
def sample_df(df, col_name='family', n_sample_per_class=120, replace = False):
    """
    samples the dataframe based on a column, duplicates only if the 
number of initial rows < required sample size
    """
    
    samples = df.groupby(col_name)
    list_cls = df[col_name].unique()
    df_lst = []
    for cls in list_cls:
        cls_df = samples.get_group(cls)
        if (cls_df.shape[0] < n_sample_per_class) and (replace==False):
            cls_sample = cls_df
        else:
            cls_sample = cls_df.sample(n=n_sample_per_class,replace=replace,random_state=42)
        df_lst.append(cls_sample)
      
    df_sampled = pd.concat(df_lst, sort=False)
    df_sampled = shuffle(df_sampled)
    
    return df_sampled
    
    
def plotImages(images_arr):
    """This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column."""
    
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    return
    
def print_layer_trainable(model_name):
    """prints out layer names and if they are trainable or not of a given model"""

    print('trainable : layer name')
    print('- '*30)
    for layer in model_name.layers:
      # if layer.trainable:
        print("{0}:\t{1}".format(layer.trainable, layer.name))
    
    return
    

def plot_training_history(history, metric):
    """function for plotting training history. This plots the classification accuracy and loss-values recorded during training with the Keras API."""
    
    val_metric = 'val_'+metric
    acc = history.history[metric]
    val_acc = history.history[val_metric]
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = history.epoch
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Acc.')
    plt.plot(epochs_range, val_acc, label='Validation Acc.')
    plt.legend(loc='best',)
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Training and Validation Loss')
    plt.show()
    
    
def create_res_labels_df(test_generator, test_history):
    """converts array returned by the predict generator into a dataframe of with true & predicted labels, and the image names
    test_history : ndimensional array returned by the mode.predict_generator() method
    test_generator : ImageDataGenerator Object that was used used by .predict_generator()    
    """
    
    df_test_results = pd.DataFrame()
    test_len = test_history.shape[0]
    df_test_results['y_true'] = test_generator.labels[:test_len]
    df_test_results['y_pred'] = tf.math.argmax(test_history, axis=1).numpy().ravel()
    df_test_results['image_path'] = test_generator.filepaths[:test_len]
    
    return df_test_results


def plot_confusion_matrix(df_res_labels):
    """function to plot confusion matrix heatmap"""

    y_tr = df_res_labels['y_true']
    y_pre = df_res_labels['y_pred']  

    labels = np.sort(y_tr.unique())
    conf_mat = metrics.confusion_matrix(y_tr, y_pre,labels=labels )
    df_conf = pd.DataFrame(conf_mat, columns=labels, index=labels)

    mask = np.ones(conf_mat.shape) 
    mask = (mask - np.diag(np.ones(conf_mat.shape[0]))).astype(np.bool)
    max_val = np.amax(conf_mat[mask])
    
    fig, ax = plt.subplots(figsize=(14,10))
    ano = True if df_conf.shape[0] < 100 else False
    sns.heatmap(df_conf,vmax=max_val, annot=ano, ax=ax)
    plt.show()
    print('- '*50)
        
    return

def create_test_report(test_generator, test_history):
    """Function to create whole test report uses create_res_labels function and plot_confusion_matrix function"""
    
    df_res_labels = create_res_labels_df(test_generator, test_history)
    
    print_metric_to_console = False
    lvls=['']
    
    metrics_dict = {}
    
    n_samples = df_res_labels.shape[0]
    print('.'*50)
    print('showing test metrics for {} samples'.format(n_samples))
    print('`'*50)
    
    lvl_metrics_dict = {}
    for lvl in lvls:
        y_tr = df_res_labels['y_true' + lvl]
        y_pre = df_res_labels['y_pred' + lvl]  
    
        lvl_metrics_dict = {}
        
        # Macro / Micro Driven Metrics
        for avg in ['macro', 'micro']:
            
            met_name = 'precision' + ('_'+ avg)    
            res = metrics.precision_score(y_tr, y_pre, average=avg)
            lvl_metrics_dict[met_name] = res
            
            met_name = 'f1' + ('_'+ avg)    
            res = metrics.f1_score(y_tr, y_pre, average=avg)
            lvl_metrics_dict[met_name] = res
            
            met_name = 'recall' + ('_'+ avg)    
            res = metrics.recall_score(y_tr, y_pre, average=avg)
            lvl_metrics_dict[met_name] = res
            
        met_name = 'accuracy'    
        res = metrics.accuracy_score(y_tr, y_pre)
        lvl_metrics_dict[met_name] = res
        
        metrics_dict[lvl] = lvl_metrics_dict
    
    df_test_results = pd.DataFrame(metrics_dict).sort_values(by=lvls, ascending=False)
    df_test_results=df_test_results.reindex(columns=lvls)
    
    print(df_test_results)
    print('- '*70)
    
    plot_confusion_matrix(df_res_labels)
    
    return df_res_labels
  
def predict(input_shape, model,  image_path):
    """This function loads the image from given image_path. resize it depending on input_shape given 
    based on model. Do prediction on the model"""
    
    # Load and resize the image using PIL.
    img = PIL.Image.open(image_path)
    print('input_shape: ', input_shape)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    # Plot the image.
    plt.imshow(img_resized)
    plt.show()

    # Convert the PIL image to a numpy-array with the proper shape.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Use the ResNet50 model to make a prediction.
    # This outputs an array with 1000 numbers corresponding to
    # the classes of the ImageNet-dataset.
    pred = model.predict(img_array)
    
    # Decode the output of the ResNet50 model.
    pred_decoded = decode_predictions(pred)[0]

    # Print the predictions.
    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))
        
    return 
