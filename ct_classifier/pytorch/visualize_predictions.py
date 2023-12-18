# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:56:20 2023

@author: jarre
"""
import os
import yaml
import argparse
import shutil
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
from eval_metrics import predict, conf_table
from sklearn.metrics import confusion_matrix

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

conf_matrix = confusion_matrix(all_true, all_preds)

conf_tab = conf_table(conf_matrix, Y_ordered)

conf_tab.columns = short_Y_ordered
conf_tab.index = short_Y_ordered

threshold = 0.1

vis_dict = {}

# Iterate through each row
for row_label, row in conf_tab.iterrows():
    # Find columns and values > threshold
    cols_above_threshold = row[row > threshold].index.tolist()
    values_above_threshold = row[row > threshold].tolist()
    
    # Add entries to the result_dict
    vis_dict[row_label] = [cols_above_threshold, values_above_threshold]

vis_path = os.path.join(
    data_root,
    'eval',
    'vis/vis.yaml')

with open(vis_path, 'w') as file:
    yaml.dump(vis_dict, file)


val_labs = meta[short_labels]
pred_labs = [0] * len(all_preds)
for i in labelIndex:
    idx = np.where(all_preds == i)[0]
    for j in idx:
        pred_labs[j] = short_Y_ordered[i]

pred_labs = np.array(pred_labs)

file_names = cfg['file_name']
for g_true in vis_dict:
    for pred in vis_dict[g_true][0]:
        idx = np.where((val_labs == g_true) & (pred_labs == pred))[0]
        idx = np.random.choice(idx, size=min(len(idx), 5), replace=False)
        files = meta[file_names][idx]
        # Create the target directory if it doesn't exist
        target_dir = os.path.join(data_root, 'eval/vis', g_true)
        
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        
        target_dir = os.path.join(target_dir, pred)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        
        for file in files:
            source_file_path = os.path.join(data_root, 'AllPhotosJPG', file)
            target_file_path = os.path.join(target_dir, file)
            shutil.copy(source_file_path, target_file_path)

        




