# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:38:49 2023

@author: blair
"""

import os

import numpy as np
import pandas as pd
from pySankey.sankey import sankey

os.chdir(r"C:\Carabid_Data\CV-eDNA\splits\order")

df = pd.read_csv("ML_DNABias.csv", sep = ",")

taxaorder = np.array(["Phylum",
              "Subphylum",
              "Class",
              "Subclass",
              "Superorder",
              "Order",
              "Suborder",
              "Infraorder",
              "Superfamily",
              "Family",
              "Subfamily",
              "Genus",
              "Species"])

rightLabels = [x in df["New"].values for x in taxaorder]
rightLabels = taxaorder[rightLabels]
rightLabels = rightLabels.tolist()

leftLabels = [x in df["Original"].values for x in taxaorder]
leftLabels = taxaorder[leftLabels]
leftLabels = leftLabels.tolist()

leftgap = 0.275
rightgap = 0.1

sankey(left = df["Original"], 
       right = df["New"],
       leftLabels = leftLabels,
       rightLabels = rightLabels,
       fontsize = 12,
       leftgap = leftgap,
       rightgap = rightgap)
