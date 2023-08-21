# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:37:09 2023

@author: jarre
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from train import create_dataloader, load_model       # NOTE: since we're using these functions across files, it could make sense to put them in e.g. a "util.py" script.
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc, confusion_matrix


def predict(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
   # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
   
    all_preds = []
    all_probs = []
    all_true = []
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model (data)
            pred_label = torch.argmax(prediction, dim=1)
            
            all_probs.append(prediction)
            all_preds.append(pred_label)
            all_true.append(labels)


    all_preds = torch.cat(all_preds, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_true = torch.cat(all_true, dim=0)
    
    all_preds = all_preds.cpu()
    all_probs = all_probs.cpu()
    all_true = all_true.cpu()
    
    all_preds = all_preds.numpy()
    all_probs = all_probs.numpy()
    all_true = all_true.numpy()
    
    return all_true, all_preds, all_probs
    

def conf_table(conf_matrix, Y, prop = True):
    """
    Creates a confusion matrix as a pandas data frame pivot table

    Parameters:
    - conf_matrix (Array): The standard confusion matrix output from sklearn
    - Y (Array): The unique labels of the classifier. Will be used as conf_table labels
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
        conf_df['prop'] = conf_df['Count'] / conf_df.groupby('Reference')['Count'].transform('sum')
        
        # Create conf_table
        conf_table = conf_df.pivot_table(index='Reference', columns='Prediction', values='prop', fill_value=0)
        
    else:
        # Create conf_table
        conf_table = conf_df.pivot_table(index='Reference', columns='Prediction', values='Count', fill_value=0)
    
    
    return conf_table


def save_conf(cfg, table, Y):
    """
    Plots a confusion matrix

    Parameters:
    - cfg (dict): The config yaml
    - table (DataFrame): The conf_table
    - Y (list): The class labels to be plotted on the conf_tab

    Returns:
    Saves plot to directory
    """
    
    data_root = cfg['data_root']
    
    custom_gradient = ["#201547", "#00BCE1"]
    n_bins = 100  # Number of bins for the gradient

    custom_cmap = LinearSegmentedColormap.from_list("CustomColormap", custom_gradient, N=n_bins)

    plt.figure(figsize=(16, 12))
    sns.set(font_scale=1)
    sns.set_style("white")
    
    ax = sns.heatmap(table, cmap=custom_cmap, cbar_kws={'label': 'Proportion'})
    
    # Customize the axis labels and ticks
    ax.set_xlabel("Predicted", fontsize=20)
    ax.set_ylabel("Actual", fontsize=20)
    ax.set_xticks(np.arange(len(Y)) + 0.5)
    ax.set_yticks(np.arange(len(Y)) + 0.5)
    ax.set_xticklabels(Y, fontsize=12)
    ax.set_yticklabels(Y, rotation=0, fontsize=12)
    
    # Add annotation
    ax.annotate("Predicted", xy=(0.5, -0.2), xytext=(0.5, -0.5), ha='center', va='center',
                 textcoords='axes fraction', arrowprops=dict(arrowstyle="-", lw=1))
    
    
    # Customize the appearance directly on the Axes object
    ax.set_xticklabels(Y, rotation=45, ha='right')
    
    # Save the plot
    conf_path = os.path.join(
        data_root,
        'eval',
        'plots/conf.png')
    
    plt.savefig(conf_path)


def main():

    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='../configs/exp_resnet18.yaml')
    args = parser.parse_args()
    
    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    
    
    # setup entities
    dl_test = create_dataloader(cfg, split='test')
    
    # load model
    model = load_model(cfg)
    
    data_root = cfg['data_root']
    
    # load annotation file
    annoPath = os.path.join(
        data_root,
        cfg["annotate_root"],
        'valid.csv'
    )
    meta = pd.read_csv(annoPath)
    
    class_labels = cfg['class_labels']
    Y = meta[class_labels]
    Y = Y.unique()
    encoder = LabelEncoder()
    encoder.fit(Y)
    labelIndex = encoder.transform(Y)
    
    short_labels = cfg['short_labels']
    short_Y = meta[short_labels]
    short_Y = short_Y.unique()
    
    Y_ordered = sorted(Y)
    short_Y_ordered = [0] * len(short_Y)
    
    for i in labelIndex:
        short_Y_ordered[labelIndex[i]] = short_Y[i]
    
    all_true, all_preds, all_probs = predict(cfg, dl_test, model[0])
    
    report = classification_report(all_true, 
                                   all_preds, 
                                   target_names=short_Y_ordered,
                                   output_dict=True)

    report_path = os.path.join(
        data_root,
        'eval',
        'metrics/eval.yaml')
    with open(report_path, 'w') as file:
        yaml.dump(report, file)
    
    conf_matrix = confusion_matrix(all_true, all_preds)
    
    conf_tab = conf_table(conf_matrix, Y_ordered)
    
    save_conf(cfg, conf_tab, short_Y_ordered)

    # That's all, folks!
        


if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()


