# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:37:09 2023

@author: jarre
"""

import yaml
import argparse
from train import create_dataloader, load_model       # NOTE: since we're using these functions across files, it could make sense to put them in e.g. a "util.py" script.


parser = argparse.ArgumentParser(description='Train deep learning model.')
parser.add_argument('--config', help='Path to config file', default='../configs/exp_resnet18.yaml')
args = parser.parse_args()

# load config
print(f'Using config "{args.config}"')
cfg = yaml.safe_load(open(args.config, 'r'))


# setup entities
dl_test = create_dataloader(cfg, split='test')

# load model
model = load_model(cfg)

print(model)