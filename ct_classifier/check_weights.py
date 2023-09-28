# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:07:31 2023

@author: blair
"""

tab_list = []
lab_list = []

with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
    for idx, (tab, imgs, labels) in enumerate(dl_test):
        tab_list.append(tab)
        lab_list.append(labels)


tab_out = model.tabular_model(tab_cat)
res_out = model.resnet_model(imgs)
cat = torch.cat((tab, test), dim=1)
classify = model.classifier(test_cat)
f_pass = model.forward(tab, imgs)
class_weight = model.classifier.weight


