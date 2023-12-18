# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:48:25 2023

@author: blair
"""

import os
os.chdir(r"C:\Users\blair\OneDrive - UBC\CV-eDNA-Hybrid\ct_classifier")

import math
import yaml
import copy
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
from util_tf import hierarchy, hierarchy_pred, conf_table, plt_conf


def main():
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='../configs')
    parser.add_argument('--exp', help='Experiment name', default='exp_resnet18_37141')
    args = parser.parse_args()
    
    # load config
    print(f'Using config "{args.config}"')
    cfg_path = os.path.join(
        args.config,
        args.exp)
    cfg = yaml.safe_load(open(cfg_path + ".yaml"))
        
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
    
    hierarchy_long, hierarchy_short = hierarchy(Y_ordered)
    named_true_long, named_true_short = hierarchy_pred(all_true, hierarchy_long, hierarchy_short)
    
    report_structure = {"Phylum": {},
                  "Class": {},
                  "Order": {},
                  "Family": {},
                  "Genus": {},
                  "Species": {}}
    
    report = [report_structure.copy() for _ in range(5)]
    average_recall = [report_structure.copy() for _ in range(5)]
    
    probs = list(range(5))
    t3_acc = list(range(5))
    predicted_classes = list(range(5))
    named_pred_long = list(range(5))
    named_pred_short = list(range(5))
    
    for i in range(len(cfg['seed'])):
        # load model
        model = tf.keras.models.load_model(f'{experiment}_{seed[i]}.h5')
        
        probs[i] = model.predict(test_generator)
        predicted_classes[i] = tf.argmax(probs[i], axis=1)
        predicted_classes[i] = predicted_classes[i].numpy()
        
        named_pred_long[i], named_pred_short[i] = hierarchy_pred(predicted_classes[i], hierarchy_long, hierarchy_short)
        
        top3_indices = np.argsort(probs[i], axis=1)[:, -3:]
        t3_class = np.any(top3_indices == all_true[:, np.newaxis], axis=1)
        t3_acc[i] = np.mean(t3_class)
        
        for level in report[i]:
            report[i][level] = classification_report(named_true_short[level],
                                  named_pred_short[i][level],
                                  output_dict=True,
                                  zero_division=1)
            recall_values = [report[i][level][key]['recall'] for key in set(named_true_short[level])]
            average_recall[i][level] = sum(recall_values) / len(recall_values)
    
    accuracy_values = [d["Species"]['accuracy'] for d in report]
    recall = [average_recall[i]["Species"] for i in range(5)]
    np.mean(accuracy_values)
    np.std(accuracy_values)
    np.mean(recall)
    np.std(recall)
    np.mean(t3_acc)
    np.std(t3_acc)

    # Mask specific path
    met_path = os.path.join(
        data_root,
        'eval',
        f'{args.exp}/metrics')
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
        f'{args.exp}/plots')
    os.makedirs(conf_path, exist_ok=True)
    
    for level in conf_tab:
        pred_long = []
        for i in named_pred_long:
            pred_long.extend(i[level])
        conf_matrix = confusion_matrix(named_true_long[level]*5,
                              pred_long,
                              labels = sorted(list(set(hierarchy_long[level]))))
        long = pd.Series(hierarchy_long[level])
        long = long.unique()
        conf_tab[level] = conf_table(conf_matrix, long)
        short = pd.Series(hierarchy_short[level])
        short = short.unique()
        avg_report = classification_report(named_true_long[level]*5,
                              pred_long,
                              output_dict=True,
                              zero_division=1)        
        figure, n = plt_conf(conf_tab[level], short, avg_report, level)
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


