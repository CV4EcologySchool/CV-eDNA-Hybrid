# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:42:36 2023

@author: blair
"""
import os
import random
import numpy as np
import tensorflow as tf

def init_seed(seed):
    
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #tf.config.experimental.enable_op_determinism()
    
