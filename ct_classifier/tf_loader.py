# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:20:47 2023

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
        """
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        """

        # Where
        self.data_root = cfg['data_root']
        self.num_class = cfg['num_classes']
        self.img_size = cfg['image_size']
        self.batch_size = cfg['batch_size']
        self.seed = cfg['seed']
        self.split = split

        # Load annotation file
        anno_path = os.path.join(
            self.data_root,
            cfg["annotate_root"],
            'train.csv' if self.split == 'train' else 'valid.csv'
        )

        train_path = os.path.join(
            os.path.dirname(anno_path),
            'train.csv'
        )

        meta = pd.read_csv(anno_path)
        train = pd.read_csv(train_path)

        class_labels = cfg['class_labels']
        Y_train = train[class_labels]
        encoder = LabelEncoder()
        encoder.fit(Y_train)

        Y = meta[class_labels]
        label_index = encoder.transform(Y)
        encoded_Y = np_utils.to_categorical(label_index)

        file_name = cfg['file_name']
        img_file_names = meta[file_name].tolist()

        self.data = list(zip(img_file_names, encoded_Y))

    def create_tf_dataset(self):
        """
            Create a TensorFlow dataset.
        """

        data = Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(self.num_class), dtype=tf.float32),
            )
        )

        # Shuffle data if split == 'train'
        if self.split == 'train':
            data = data.shuffle(buffer_size=1000, seed = self.seed)
            data = data.map(
                self.data_augmentation,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
                )

        # Preprocess images and labels
        data = data.map(
            lambda img_array, label: (preprocess_input(img_array), label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # Batch data
        data = data.batch(self.batch_size)
        
        # Prefetch data to the GPU (assuming you have GPU support)
        data = data.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))

        return data

    def generator(self):
        """
            Generator function for the TensorFlow dataset.
        """
        for image_name, label in self.data:
            # Load image
            image_path = os.path.join(self.data_root, 'AllPhotosJPG', image_name)
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img)

            yield (img_array, label)
            
    def data_augmentation(self, img_array, label):
        """
            Apply data augmentation (randomly flip left-right) to the image.
        """
        img_array = tf.image.random_flip_left_right(img_array)
        return img_array, label
