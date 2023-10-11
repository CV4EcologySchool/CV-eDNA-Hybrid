# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:40:47 2023

@author: jarre
"""


from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np
import json
import os

# Sample data
with open(r"C:\Carabid_Data\CV-eDNA\splits\LKTL-37141\image_multilab_nosplit.json") as json_file:
    image_multilab = json.load(json_file)
with open(r"C:\Carabid_Data\CV-eDNA\splits\LKTL-37141\dna_multilab_nosplit.json") as json_file:
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

df_array = np.column_stack((list(image_multilab.keys()), predicted_labels_bin))
df = pd.DataFrame(df_array)
df.rename(columns={0: "event"}, inplace=True)

# Save the Pandas DataFrame as a CSV file
df.to_csv('dna_mhe.csv', index=False)

