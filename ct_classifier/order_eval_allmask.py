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
from util_order import hierarchy, hierarchy_pred, conf_table, plt_conf

parser = argparse.ArgumentParser(description='Train deep learning model.')
parser.add_argument('--config', help='Path to config file', default='../configs/exp_order_base.yaml')
parser.add_argument('--mask', help='Experiment name', default='naive')
args = parser.parse_args()

# load config
print(f'Using config "{args.config}"')
cfg = yaml.safe_load(open(args.config, 'r'))
    
experiment = cfg['experiment_name']
data_root = cfg['data_root']
seed = cfg['seed']

# setup entities
test_loader = CTDataset(cfg, split='valid')   
test_generator = test_loader.create_tf_dataset()
 
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

named_true_long = [Y_ordered[index] for index in all_true]
named_true_short = [Y_ordered[index] for index in all_true]

   
'''
Mask code begins
'''

mhePath = os.path.join(
    os.path.dirname(annoPath),
    'naive_sim.csv'
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

if(args.mask == "weighted"):
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

'''
Mask code ends
'''
   
# load model
model = tf.keras.models.load_model(f'model_states\{experiment}\{experiment}_loss_w.h5')

probs = model.predict(test_generator)
predicted_classes = tf.argmax(probs, axis=1)
predicted_classes = predicted_classes.numpy()

result_matrix = np.zeros_like(probs)
masked_pred_ohe = np.zeros_like(probs)

# Iterate through the keys in 'mhe'
for key, value_list in filtered_mhe_dict.items():
    # Find the index for the current category
    idx = event_indices[key]
    # Multiply the corresponding rows by the value list
    probs[idx, :] = probs[idx, :] * value_list
    # Assign it to a new matrix
    result_matrix[idx, :] = probs[idx, :]

masked_preds = np.argmax(result_matrix, axis = 1)
masked_pred_ohe[np.arange(len(result_matrix)), masked_preds] = 1

top3_indices = np.argsort(probs, axis=1)[:, -3:]
t3_class = np.any(top3_indices == all_true[:, np.newaxis], axis=1)
t3_acc = np.mean(t3_class)

named_pred_long = [Y_ordered[index] for index in masked_preds]
named_pred_short = [Y_ordered[index] for index in masked_preds]
    
report = classification_report(named_true_short,
                      named_pred_short,
                      output_dict=True,
                      zero_division=1)
recall_values = [report[key]['recall'] for key in set(named_true_short)]
average_recall = sum(recall_values) / len(recall_values)


# # Mask specific path
# met_path = os.path.join(
#     data_root,
#     'eval',
#     f'{args.exp}_weighted/metrics')
# os.makedirs(met_path, exist_ok=True)
# report_path = os.path.join(
#     met_path,
#     'eval.yaml')
# with open(report_path, 'w') as file:
#     yaml.dump(report, file)

# conf_path = os.path.join(
#     data_root,
#     'eval',
#     f'{args.exp}_weighted/plots')
# os.makedirs(conf_path, exist_ok=True)


conf_matrix = confusion_matrix(named_true_long,
                      named_pred_long,
                      labels = Y_ordered)

conf_tab = conf_table(conf_matrix, Y_ordered)    
figure, n = plt_conf(conf_tab, short_Y_ordered, report)
# # Save the plot
# fig_path = os.path.join(
#     conf_path,
#     f'conf_{n}_{level}.png')
       
# figure.savefig(fig_path)


#save_conf_html(cfg, conf_tab, short_Y_ordered, report)

