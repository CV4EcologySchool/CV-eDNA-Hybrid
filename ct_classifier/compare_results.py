# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:51:45 2023

@author: blair
"""

percent_sim = []
for i in range(5):
    weighted = np.array(pred_long_weighted[i]["Species"]) 
    normal = np.array(pred_long[i]["Species"]) 
    
    match = weighted == normal
    not_match = np.logical_not(match)
    
    percent_sim.append(sum(match)/len(normal))

true_long = np.array(named_true_long[level]*5)

weighted_acc = weighted == true_long
normal_acc = normal == true_long

weighted_acc_not = weighted_acc[not_match]
normal_acc_not = normal_acc[not_match]

# Initialize counts for the four categories
true_true = np.sum(normal_acc_not & weighted_acc_not)/5
true_false = np.sum(normal_acc_not & ~weighted_acc_not)/5
false_true = np.sum(~normal_acc_not & weighted_acc_not)/5
false_false = np.sum(~normal_acc_not & ~weighted_acc_not)/5

# Create a contingency table as a NumPy array
contingency_table = np.array([[true_true, true_false],
                             [false_true, false_false]])
