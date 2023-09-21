# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:46:49 2023

@author: jarre
"""

import os
os.chdir(r"C:\Users\blair\OneDrive - UBC\CV-eDNA-Hybrid\ct_classifier")

import yaml
import copy
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import trange
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
from train import create_dataloader, load_model       # NOTE: since we're using these functions across files, it could make sense to put them in e.g. a "util.py" script.
from eval_metrics import predict, hierarchy, hierarchy_pred, conf_table, plt_conf, save_conf_html
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from dataset import CTDataset   # Leave this, it helps for some reason

def main():
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='../configs')
    parser.add_argument('--exp', help='Experiment name', default='exp_resnet18_37141')
    parser.add_argument('--mask', help='Experiment name', default='naive')
    args = parser.parse_args()
    
    # load config
    print(f'Using config "{args.config}"')
    cfg_path = os.path.join(
        args.config,
        args.exp)
    cfg = yaml.safe_load(open(cfg_path + ".yaml"))
        
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
    
    all_true, all_preds, all_probs = predict(cfg, dl_test, model[0])
    
    all_probs = torch.tensor(all_probs)
    softmax = torch.nn.Softmax(dim=1)
    all_probs = softmax(all_probs)
    all_probs = all_probs.cpu()
    all_probs = all_probs.numpy()
    
    '''
    Mask code begins
    '''
    
    mhePath = os.path.join(
        os.path.dirname(annoPath),
        f'{args.mask}.csv'
    )
    # Load CSV into a pandas DataFrame
    mhe_df = pd.read_csv(mhePath)
    
    events = mhe_df["event"]
    #mhe_df.rename(columns={'Canthon.viridis': 'Canthon viridis'}, inplace=True)
    #mhe_df = mhe_df[short_Y_ordered]
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
    # Create a new matrix to store the results
    result_matrix = np.zeros_like(all_probs)
    
    # Iterate through the keys in 'mhe'
    for key, value_list in filtered_mhe_dict.items():
        # Find the index for the current category
        idx = event_indices[key]
        
        # Multiply the corresponding rows by the value list
        result_matrix[idx, :] = all_probs[idx, :] * value_list
    
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


