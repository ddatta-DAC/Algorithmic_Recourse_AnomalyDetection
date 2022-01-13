import argparse
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('./..')
from model_AD_1 import AD_model_container, MEAD
import AD_train_executor

parser = argparse.ArgumentParser(description='Train anomaly detection model (MEAD)')
parser.add_argument('--dir', type = str,  help='Which dataset ? us__import{1,2...}' ) 
args = parser.parse_args()
DIR = args.dir

AD_train_executor.setup_config(DIR)
AD_train_executor.train_AD_models()
AD_train_executor.calculate_thresholds()
    