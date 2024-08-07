# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 08:22:17 2024

@author: blair
"""

#from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy as np
import json
import os

# Sample data
with open(r"C:\Carabid_Data\CV-eDNA\splits\order\image_multilab_order_train.json") as json_file:
    image_multilab = json.load(json_file)
with open(r"C:\Carabid_Data\CV-eDNA\splits\order\dna_multilab_order_train.json") as json_file:
    dna_multilab = json.load(json_file)

ground_truth = [value for value in image_multilab.values()]
predicted_labels = [value for value in dna_multilab.values()]

# Get unique labels
all_labels = sorted(list(set(label for labels in ground_truth + predicted_labels for label in labels)))

# Create label-to-index and index-to-label mappings
label_to_index = {label: index for index, label in enumerate(all_labels)}
index_to_label = {index: label for label, index in label_to_index.items()}

# Convert labels to binary arrays
def label_to_binary(label_list):
    binary_array = np.zeros(len(all_labels))
    for label in label_list:
        binary_array[label_to_index[label]] = 1
    return binary_array

ground_truth_bin = np.array([label_to_binary(labels) for labels in ground_truth])
predicted_labels_bin = np.array([label_to_binary(labels) for labels in predicted_labels])

# Calculate precision and recall for each label
precisions = []
recalls = []
specificities = []

for label_index in range(len(all_labels)):
    precision = precision_score(ground_truth_bin[:, label_index], 
                                predicted_labels_bin[:, label_index], 
                                zero_division = np.nan)
    recall = recall_score(ground_truth_bin[:, label_index], 
                          predicted_labels_bin[:, label_index])
    
    conf_matrix = confusion_matrix(ground_truth_bin[:, label_index], 
                                   predicted_labels_bin[:, label_index])
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    specificity = TN / (TN + FP)
    
    precisions.append(precision)
    recalls.append(recall)
    specificities.append(specificity)

# Calculate precision and recall for each label
label_precision_recall = {}

for label_index, label in index_to_label.items():
    precision = precision_score(ground_truth_bin[:, label_index], 
                                predicted_labels_bin[:, label_index],
                                zero_division = 1)
    recall = recall_score(ground_truth_bin[:, label_index], 
                          predicted_labels_bin[:, label_index])
    
    conf_matrix = confusion_matrix(ground_truth_bin[:, label_index], 
                                   predicted_labels_bin[:, label_index])
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    specificity = TN / (TN + FP)
    
    label_precision_recall[label] = {'precision': precision, 
                                     'recall': recall,
                                     'specificity': specificity}

np.nanmean(precisions)
np.nanmean(recalls)
np.nanmean(specificities)

os.chdir(r"C:\Carabid_Data\CV-eDNA\splits\order")
json_filename = "dna_pr.json"
with open(json_filename, "w") as json_file:
    json.dump(label_precision_recall, json_file, indent=4)
