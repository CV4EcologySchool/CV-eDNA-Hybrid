# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:36:48 2023

@author: blair
"""

from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Dropout, Activation
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Input
import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Users\blair\OneDrive - UBC\CV-eDNA-Hybrid\ct_classifier")

import argparse
import yaml
from util_tf import init_seed
from TF_Loader import CTDataset



parser = argparse.ArgumentParser(description='Train deep learning model.')
parser.add_argument('--config', help='Path to config file', default='../configs/exp_resnet18_37141_concat.yaml')
args = parser.parse_args()

# load config
print(f'Using config "{args.config}"')
cfg = yaml.safe_load(open(args.config, 'r'))

seed = cfg["seed"]
batch_size = cfg["batch_size"]

init_seed(seed)

# Initialize the dataset
train_loader = CTDataset(cfg, split='train')
valid_loader = CTDataset(cfg, split='valid')

# Create a TensorFlow dataset
train_data = train_loader.create_tf_dataset(batch_size= batch_size, shuffle_buffer_size=1000, seed = seed)
valid_data = valid_loader.create_tf_dataset(batch_size= batch_size, shuffle_buffer_size=1000, seed = seed)

ncol = cfg["num_col"]
num_class = cfg["num_classes"]

# Define simple ANN for tabular data
inputs = Input(shape = (ncol,))
annx = Dense(128, activation = 'relu')(inputs)
ann = Model(inputs, annx)

# Define ResNet for image data
base_model = ResNet50(include_top = False, weights = 'imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
resnet = Model(inputs = base_model.input, outputs = x)

for layer in base_model.layers:
    layer.trainable = False
    
concat = concatenate([ann.output, resnet.output])

combined = Dense(128)(concat)
combined = BatchNormalization()(combined)
combined = Activation('relu')(combined)
combined = Dropout(0.3)(combined)
combined = Dense(num_class, activation = "softmax")(combined)
model = Model(inputs = [ann.input, resnet.input], outputs = combined)
    
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_data,
    epochs=10, 
    verbose = 1,
    validation_data = valid_data)

validdf = pd.read_csv("shufflevalidlitl.csv")
validX = validdf.drop(["AllTaxa"], axis = 1)
validY = validdf["AllTaxa"]
encoder = LabelEncoder()
encoder.fit(validY)
encoded_validY = encoder.transform(validY)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_validY = np_utils.to_categorical(encoded_validY)

preds = model.predict([validX, validimages])