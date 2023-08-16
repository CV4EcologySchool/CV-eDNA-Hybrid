# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:07:33 2023

@author: jarre
"""

import os
import sys
import shutil
import pandas as pd
from progress_bar import progress_bar

# Paths
csv_file_path = "C:/Users/jarre/ownCloud/CV-eDNA/invertmatch.csv"
source_image_dir = "C:/Users/jarre/ownCloud/CV-eDNA/AllPhotosJPG"
target_base_dir = "C:/Users/jarre/ownCloud/CV-eDNA/splits/LKTL_indiscriminate"

# Read the CSV file using pandas
data = pd.read_csv(csv_file_path)

i = 0
total = data.shape[0]
# Iterate through each row in the CSV and copy images to their respective folders
for index, row in data.iterrows():
    image_filename = row["Label"]
    split_category = row["Split"]
    
    source_image_path = os.path.join(source_image_dir, image_filename)
    target_image_dir = os.path.join(target_base_dir, split_category)
    target_image_path = os.path.join(target_image_dir, image_filename)
    
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_image_dir):
        os.makedirs(target_image_dir)
    
    # Copy the image file
    shutil.copy(source_image_path, target_image_path)
    
    progress_bar(i + 1, total)
    i += 1

print("\nImage copying completed.")
