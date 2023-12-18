# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:25:34 2023

@author: blair
"""

# Iterate through the TensorFlow dataset
for batch in train_data.take(1):
    Data, labels = batch
    break

# Iterate through the TensorFlow dataset
for batch in valid_data.take(1):
    Data, labels = batch
    break
