# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:02:06 2023

@author: blair
"""

import torch
import torch.nn as nn
from torchvision.models import resnet


class CustomResNet18(nn.Module):

    def __init__(self):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        super(CustomResNet18, self).__init__()

        self.feature_extractor = resnet.resnet18(pretrained=True)       # "pretrained": use weights pre-trained on ImageNet

        # replace the very last layer from the original, 1000-class output
        # ImageNet to a new one that outputs num_classes
        last_layer = self.feature_extractor.fc                          # tip: print(self.feature_extractor) to get info on how model is set up
        self.in_features = last_layer.in_features                            # number of input dimensions to last (classifier) layer
        self.feature_extractor.fc = nn.Identity()                       # discard last layer...

    

    def forward(self, x):
        '''
            Forward pass. Here, we define how to apply our model. It's basically
            applying our modified ResNet-18 on the input tensor ("x") and then
            apply the final classifier layer on the ResNet-18 output to get our
            num_classes prediction.
        '''
        # x.size(): [B x 3 x W x H]
        features = self.feature_extractor(x)    # features.size(): [B x 512 x W x H]

        return features
    
# Concatenated Model
class ConcatenateModel(nn.Module):
    def __init__(self, resnet_model, num_classes, num_col):
        super(ConcatenateModel, self).__init__()
        self.resnet_model = resnet_model
        self.classifier = nn.Linear(num_col + resnet_model.in_features, num_classes)
        
    def forward(self, x_tabular, x_images):
        x_images = self.resnet_model(x_images)
        x_combined = torch.cat((x_tabular, x_images), dim=1)
        output = self.classifier(x_combined)
        return output



