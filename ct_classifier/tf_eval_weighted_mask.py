# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:24:49 2023

@author: blair
"""


import os
os.chdir(r"C:\Users\blair\OneDrive - UBC\CV-eDNA-Hybrid\ct_classifier")

import yaml
import copy
import json
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
# from tqdm import trange
# from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
# import plotly.graph_objects as go
# from eval_metrics import predict, hierarchy, hierarchy_pred, conf_table, plt_conf, save_conf_html
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tf_loader import CTDataset   # Leave this, it helps for some reason

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
    
    ax = sns.heatmap(table, cmap=custom_cmap, cbar_kws={'label': 'Proportion'})
    
    ax.set_title(f"Level = {level} ; Accuracy = {accuracy} ; Recall = {recall}; n = {n}", fontsize=24)
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
    
    return plt.gcf(), n




def main():
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='../configs')
    parser.add_argument('--exp', help='Experiment name', default='exp_resnet18_37141')
    parser.add_argument('--mask', help='Mask name', default='weighted')
    args = parser.parse_args()
    
    # load config
    print(f'Using config "{args.config}"')
    cfg_path = os.path.join(
        args.config,
        args.exp)
    cfg = yaml.safe_load(open(cfg_path + ".yaml"))
        
    experiment = cfg['experiment_name']
    data_root = cfg['data_root']
    
    # setup entities
    test_loader = CTDataset(cfg, split='valid')   
    test_generator = test_loader.create_tf_dataset()
    
    # load model
    model = tf.keras.models.load_model(f'{experiment}.h5')
    
    probs = model.predict(test_generator)
    predicted_classes = tf.argmax(probs, axis=1)
    predicted_classes = predicted_classes.numpy()
    
    # load annotation file
    annoPath = os.path.join(
        data_root,
        cfg["annotate_root"],
        'valid.csv'
    )
    
    trainPath = os.path.join(
        os.path.dirname(annoPath),
        'train.csv'
    )
    train  = pd.read_csv(trainPath)      
    meta = pd.read_csv(annoPath)
    
    class_labels = cfg['class_labels']
    Y = train[class_labels]
    Y = Y.unique()
    encoder = LabelEncoder()
    encoder.fit(Y)
    labelIndex = encoder.transform(Y)
    
    short_labels = cfg['short_labels']
    short_Y = train[short_labels]
    short_Y = short_Y.unique()
    
    Y_ordered = sorted(Y)
    short_Y_ordered = [0] * len(short_Y)
    
    for i in labelIndex:
        short_Y_ordered[labelIndex[i]] = short_Y[i]
    
    all_true = []
    for idx, (data, labels) in enumerate(test_generator):

        all_true.append(labels)
    
    all_true = tf.concat(all_true, axis=0)
    all_true = tf.argmax(all_true, axis=1)
    all_true = all_true.numpy()
    
    
    '''
    Mask code begins
    '''
    
    mhePath = os.path.join(
        os.path.dirname(annoPath),
        'naive.csv'
    )
    # Load CSV into a pandas DataFrame
    mhe_df = pd.read_csv(mhePath)
    
    events = mhe_df["event"]
    mhe_df = mhe_df.drop("event", axis = 1)
    mhe_df.columns = range(mhe_df.shape[1])
    mhe_df = mhe_df.applymap(lambda x: 1 if x > 0 else x)
    mhe_df['event'] = events
    
    mhe_dict = {}
    for index, row in mhe_df.iterrows():
        category = row['event']
        values = row[:-1].tolist()
        mhe_dict[category] = values
    
    
    # Create a dictionary to store the index mapping for each category
    event_indices = {event: [] for event in set(meta['Event'])}
    for i, event in enumerate(meta['Event']):
        event_indices[event].append(i)
        
    filtered_mhe_dict = {key: value for key, value in mhe_dict.items() if key in event_indices}
    
    dnaprPath = os.path.join(
        os.path.dirname(annoPath),
        'dna_pr.json'
    )
    # Read JSON data from file
    with open(dnaprPath) as json_file:
        dna_pr = json.load(json_file)
    
    for key in filtered_mhe_dict:
        for j in range(len(filtered_mhe_dict[key])):
            if filtered_mhe_dict[key][j]:
                filtered_mhe_dict[key][j] = dna_pr[Y_ordered[j]]['precision']
            else:
                filtered_mhe_dict[key][j] = 1 - dna_pr[Y_ordered[j]]['recall']
    
    # Create a new matrix to store the results
    result_matrix = np.zeros_like(probs)
    
    # Iterate through the keys in 'mhe'
    for key, value_list in filtered_mhe_dict.items():
        # Find the index for the current category
        idx = event_indices[key]
        
        # Multiply the corresponding rows by the value list
        result_matrix[idx, :] = probs[idx, :] * value_list
    
    masked_preds = np.argmax(result_matrix, axis = 1)
    '''
    Mask code ends
    '''

    hierarchy_long, hierarchy_short = hierarchy(Y_ordered)
    
    named_true_long, named_true_short = hierarchy_pred(all_true, hierarchy_long, hierarchy_short)
    named_pred_long, named_pred_short = hierarchy_pred(masked_preds, hierarchy_long, hierarchy_short)
    
    
    report = {"Phylum": {},
                  "Class": {},
                  "Order": {},
                  "Family": {},
                  "Genus": {},
                  "Species": {}}
    
    for level in report:
        report[level] = classification_report(named_true_short[level],
                              named_pred_short[level],
                              output_dict=True,
                              zero_division=1)

    # Mask specific path
    met_path = os.path.join(
        data_root,
        'eval',
        f'{args.exp}_{args.mask}/metrics')
    os.makedirs(met_path, exist_ok=True)
    report_path = os.path.join(
        met_path,
        'eval.yaml')
    with open(report_path, 'w') as file:
        yaml.dump(report, file)
    
    conf_tab= {"Phylum": [],
                  "Class": [],
                  "Order": [],
                  "Family": [],
                  "Genus": [],
                  "Species": []}
    
    conf_path = os.path.join(
        data_root,
        'eval',
        f'{args.exp}_{args.mask}/plots')
    os.makedirs(conf_path, exist_ok=True)
    for level in conf_tab:
        conf_matrix = confusion_matrix(named_true_long[level],
                              named_pred_long[level],
                              labels = sorted(list(set(hierarchy_long[level]))))
        long = pd.Series(hierarchy_long[level])
        long = long.unique()
        conf_tab[level] = conf_table(conf_matrix, long)
        short = pd.Series(hierarchy_short[level])
        short = short.unique()
        figure, n = plt_conf(conf_tab[level], short, report[level], level)
        # Save the plot
        fig_path = os.path.join(
            conf_path,
            f'conf_{n}_{level}.png')
               
        figure.savefig(fig_path)
    
    
    #save_conf_html(cfg, conf_tab, short_Y_ordered, report)



if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()


