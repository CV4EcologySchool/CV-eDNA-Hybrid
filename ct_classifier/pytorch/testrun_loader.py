# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:39:10 2023

@author: blair
"""
import tensorflow as tf
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical

class CTDataset:

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        # Where 
        self.data_root = cfg['data_root']
        self.num_class = cfg['num_classes']
        self.img_size = cfg['image_size']
        self.batch_size = cfg['batch_size']
        self.seed = cfg['seed']
        
        self.num_col = cfg['num_col']
        
        self.split = split
        
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
        encoded_Y = to_categorical(label_index)
        
        file_name = cfg['file_name']
        img_file_names = meta[file_name].tolist()
        
        self.data = list(zip(X, img_file_names, encoded_Y))

    def create_tf_dataset(self):
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
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            )
        )

        # Shuffle and batch the dataset
        if self.split == 'train':
            data = data.shuffle(buffer_size = 1000, seed = self.seed)
            data = data.map(
                self.data_augmentation,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
        
        # Preprocess images and labels
        data = data.map(
            lambda Data, labels: (self.preprocess_fn(Data), labels),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # Batch data
        data = data.batch(self.batch_size)
        
        # Prefetch data to the GPU (assuming you have GPU support)
        data = data.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
        
        return data

    def generator(self):
        '''
            Generator function for the TensorFlow dataset.
        '''
        for X, image_name, label in self.data:
            # Load image
            image_path = os.path.join(self.data_root, 'AllPhotosJPG', image_name)
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3)  # Assuming RGB images
            img = tf.image.resize(img, self.img_size)
            img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float32

            yield ((X, img), label)
    
    def data_augmentation(self, Data, label):
        """
            Apply data augmentation (randomly flip left-right) to the image.
        """
        X, img = Data
        img = tf.image.random_flip_left_right(img)
        return (X, img), label
    
    def preprocess_fn(self, Data):
        X, img = Data
        img = preprocess_input(img)  # Apply preprocessing to img
        return (X, img)  # Return the updated structure


