# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:48:26 2023

@author: blair
"""

from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tensorflow.keras.layers import Input
import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Users\blair\OneDrive - UBC\CV-eDNA-Hybrid\ct_classifier")

import json
import argparse
import yaml
from util_tf import init_seed
from tf_loader import CTDataset

#import sys

parser = argparse.ArgumentParser(description='Train deep learning model.')
parser.add_argument('--config', help='Path to config file', default='../configs/exp_resnet18_37141.yaml')
parser.add_argument('--seed', help='Seed index', type=int, default = 0)
args = parser.parse_args()

# load config
print(f'Using config "{args.config}"')
cfg = yaml.safe_load(open(args.config, 'r'))

print(args.seed)

cfg["seed"] = cfg["seed"][args.seed]
seed = cfg["seed"]
batch_size = cfg["batch_size"]
num_class = cfg["num_classes"]
experiment = cfg["experiment_name"]

output_file = f'{experiment}.txt'

#sys.stdout = open(output_file, "w")

init_seed(seed)

# Initialize the dataset
train_loader = CTDataset(cfg, split='train')
valid_loader = CTDataset(cfg, split='valid')

# Create a TensorFlow dataset
train_data = train_loader.create_tf_dataset()
valid_data = valid_loader.create_tf_dataset()

# Define ResNet for image data
base_model = ResNet50(include_top = False, weights = 'imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
predict = Dense(num_class, activation = "softmax")(x)
model = Model(inputs = base_model.input, outputs = predict)

for layer in base_model.layers:
    layer.trainable = False
      
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=1)
checkpoint = ModelCheckpoint(f'{experiment}_{seed}.h5', monitor='val_loss', save_best_only=True)

# min_epochs = 50
# patience = 10

epochs = cfg['num_epochs']

history = model.fit(train_data,
                    epochs = epochs, 
                    verbose = 1,
                    validation_data = valid_data,
                    callbacks = [checkpoint])

best_epoch = np.argmin(history.history['val_loss']) + 1

with open(f'{experiment}_{seed}_{best_epoch}.json', 'w') as json_file:
    json.dump(history.history, json_file)


#sys.stdout.close()
#sys.stdout = sys.__stdout__

# import matplotlib.pyplot as plt

# # Plot accuracy
# plt.figure(figsize=(8, 6))
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title(f'{experiment} Accuracy', fontsize=16)
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('Accuracy', fontsize=14)
# plt.legend(['Train', 'Validation'], loc='upper left', fontsize=12)
# plt.tick_params(axis='both', labelsize=12)
# plt.show()

# # Plot loss
# plt.figure(figsize=(8, 6))
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title(f'{experiment} Loss', fontsize=16)
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('Loss', fontsize=14)
# plt.legend(['Train', 'Validation'], loc='upper right', fontsize=12)
# plt.tick_params(axis='both', labelsize=12)
# plt.show()
