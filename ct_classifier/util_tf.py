# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:42:36 2023

@author: blair
"""
import os
import math
import copy
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def init_seed(seed):
    
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #tf.config.experimental.enable_op_determinism()
    
class EarlyMinStopping(Callback):
    def __init__(self, min_epochs, patience, monitor='val_loss'):
        super(EarlyMinStopping, self).__init__()
        self.min_epochs = min_epochs
        self.patience = patience
        self.monitor = monitor
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.min_epochs:
            return

        current_value = logs.get(self.monitor)
        if current_value is None:
            raise ValueError(f"Early stopping monitor '{self.monitor}' not found in logs.")

        if current_value < self.best:
            self.best = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_begin(self, logs=None):
        self.best = float('inf')

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Training stopped after {self.stopped_epoch + 1} epochs without improvement.")

def hierarchy(Y_ordered):
    hierarchy_long = {"Phylum": Y_ordered.copy(),
                      "Class": Y_ordered.copy(),
                      "Order": Y_ordered.copy(),
                      "Family": Y_ordered.copy(),
                      "Genus": Y_ordered.copy(),
                      "Species": Y_ordered.copy()}
    
    i = 1
    for level in hierarchy_long:
        for j in range(len(hierarchy_long[level])):
            input_string = hierarchy_long[level][j]
            parts = input_string.split("_")
            hierarchy_long[level][j] = "_".join(parts[:i])
        i += 1
    
    hierarchy_short = copy.deepcopy(hierarchy_long)
    for level in hierarchy_short:
        for j in range(len(hierarchy_short[level])):
            input_string = hierarchy_short[level][j]
            parts = input_string.split("_")
            hierarchy_short[level][j] = parts[-1]
    
    return hierarchy_long, hierarchy_short 

def hierarchy_pred(y, hierarchy_long, hierarchy_short):
   

    named_long = {"Phylum": y.copy(),
                  "Class": y.copy(),
                  "Order": y.copy(),
                  "Family": y.copy(),
                  "Genus": y.copy(),
                  "Species": y.copy()}
    
    named_short = copy.deepcopy(named_long)
    
    for level in hierarchy_long:
        named_long[level] = [hierarchy_long[level][index] for index in y]
        named_short[level] = [hierarchy_short[level][index] for index in y]
    
    return named_long, named_short

def conf_table(conf_matrix, Y, prop = True):
    """
    Creates a confusion matrix as a pandas data frame pivot table

    Parameters:
    - conf_matrix (Array): The standard confusion matrix output from sklearn
    - Y (list): The unique labels of the classifier (i.e. the classes of the output layer). Will be used as conf_table labels
    - prop (bool): Should the conf table use proportions (i.e. Recall) or total values?

    Returns:
    DataFrame: conf_table
    """
    
    # Convert conf_matrix to list
    conf_data = []
    for i, row in enumerate(conf_matrix):
        for j, count in enumerate(row):
            conf_data.append([Y[i], Y[j], count])
    
    # Convert list to 
    conf_df = pd.DataFrame(conf_data, columns=['Reference', 'Prediction', 'Count'])
    
    # If prop = True, calculate proportions
    if prop:
        conf_df['prop'] = conf_df['Count'] / (conf_df.groupby('Reference')['Count'].transform('sum') + 0.1)
        
        # Create conf_table
        conf_table = conf_df.pivot_table(index='Reference', columns='Prediction', values='prop', fill_value=0)
        
    else:
        # Create conf_table
        conf_table = conf_df.pivot_table(index='Reference', columns='Prediction', values='Count', fill_value=0)
    
    
    return conf_table

def plt_conf(table, Y, report, level = "species"):
    """
    Plots a confusion matrix

    Parameters:
    - table (DataFrame): The conf_table
    - Y (list): The class labels to be plotted on the conf_tab
    - report (dict): From sklearn.metrics.classification_report

    Returns:
    Saves plot to directory
    """
      
    accuracy = round(report["accuracy"], 3)
    recall = round(report["macro avg"]["recall"], 3)
    n = len(Y)
    
    custom_gradient = ["#201547", "#00BCE1"]
    n_bins = 100  # Number of bins for the gradient

    custom_cmap = LinearSegmentedColormap.from_list("CustomColormap", custom_gradient, N=n_bins)

    plt.figure(figsize=(16, 12))
    sns.set(font_scale=1)
    sns.set_style("white")
    
    ax = sns.heatmap(table, cmap=custom_cmap, cbar=False)
    
    ax.set_title(f"Level = {level} ; Accuracy = {accuracy} ; Recall = {recall}; n = {n}", fontsize=36)
    # Customize the axis labels and ticks
    ax.set_xlabel("Predicted", fontsize=32)
    ax.set_ylabel("Actual", fontsize=32)
    ax.set_xticks(np.arange(len(Y)) + 0.5)
    ax.set_yticks(np.arange(len(Y)) + 0.5)
    ax.set_xticklabels(Y, fontsize=12 + 12*(math.log(37/n, 10)))
    ax.set_yticklabels(Y, rotation=0, fontsize=12 + 12*(math.log(37/n, 10)))
    
    # Add annotation
    ax.annotate("Predicted", xy=(0.5, -0.2), xytext=(0.5, -0.5), ha='center', va='center',
                 textcoords='axes fraction', arrowprops=dict(arrowstyle="-", lw=1))
    
    
    # Customize the appearance directly on the Axes object
    ax.set_xticklabels(Y, rotation=45, ha='right')
    
    # Get the color bar
    cbar = ax.figure.colorbar(ax.collections[0])
    
    # Adjust color bar font size
    cbar.ax.set_ylabel('Proportion', rotation=90, fontsize=24)  # Rotate the label to 90 degrees
    cbar.ax.get_yticklabels()[0].set_verticalalignment('center')  # Adjust vertical alignment for the label
    cbar.ax.tick_params(labelsize=20) 
    
    return plt.gcf(), n

def multilabel_accuracy(y_true, y_pred):
    num_samples = len(y_true)
    correct_samples = 0

    for i in range(num_samples):
        true_labels = set(y_true[i])
        predicted_labels = set(y_pred[i])

        if len(true_labels.intersection(predicted_labels)) > 0:
            correct_samples += 1

    accuracy = correct_samples / num_samples
    return accuracy

