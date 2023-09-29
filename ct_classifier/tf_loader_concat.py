# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 18:09:14 2023

@author: blair
"""

import tensorflow as tf
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.data import Dataset
from keras.utils import np_utils

class CTDataset:

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        # Where 
        self.data_root = cfg['data_root']
        self.split = split
        self.cfg = cfg
        
        train_name = cfg["train_name"]
        val_name = cfg["val_name"]
        
        # Load annotation file
        anno_path = os.path.join(
            self.data_root,
            cfg["annotate_root"],
            f'{train_name}.csv' if self.split == 'train' else f'{val_name}.csv'
        )
        
        train_path = os.path.join(
            os.path.dirname(anno_path),
            'train.csv'
        )
        
        meta = pd.read_csv(anno_path)
        train = pd.read_csv(train_path)
        
        data_cols = range(cfg['data_cols'][0], cfg['data_cols'][1])
        
        X = meta.iloc[:, data_cols].values
        X = tf.constant(X, dtype=tf.float32)
        
        class_labels = cfg['class_labels']
        Y_train = train[class_labels]
        encoder = LabelEncoder()
        encoder.fit(Y_train)
        
        Y = meta[class_labels]
        label_index = encoder.transform(Y)
        encoded_Y = np_utils.to_categorical(label_index)
        
        file_name = cfg['file_name']
        img_file_names = meta[file_name].tolist()
        
        self.data = list(zip(X, img_file_names, encoded_Y))

    def create_tf_dataset(self, batch_size=32, shuffle_buffer_size=1000, seed = 123):
        '''
            Create a TensorFlow dataset.
        '''
        data = Dataset.from_generator(
            self.generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(None,), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                ),
                tf.TensorSpec(shape=(self.cfg['num_classes']), dtype=tf.float32),
            )
        )

        # Shuffle and batch the dataset
        if self.split == 'train':
            data = data.shuffle(shuffle_buffer_size, seed = seed)
            
        data = data.batch(batch_size)

        return data

    def generator(self):
        '''
            Generator function for the TensorFlow dataset.
        '''
        for X, image_name, label in self.data:
            # Load image
            image_path = os.path.join(self.data_root, 'AllPhotosJPG', image_name)
            img = load_img(image_path, target_size=self.cfg['image_size'])
            img_array = img_to_array(img)
            
            img_array = preprocess_input(img_array)
            
            if self.split == 'train':
                img_array = tf.image.random_flip_left_right(img_array)

            yield ((X, img_array), label)

