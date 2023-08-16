# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:27:22 2023

@author: blair
"""

import sys

def progress_bar(current, total, bar_length=40):
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    percentage = int(progress * 100)
    sys.stdout.write(f'\r[{bar}] {percentage}%')
    sys.stdout.flush()