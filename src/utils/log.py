from time import time
from datetime import datetime
from pathlib import Path
import multiprocessing
from pprint import pprint
import torch
import math
import yaml
import matplotlib.pyplot  as plt
from sklearn.metrics import auc
import logging
import logging.handlers
from time import time
import os
import sys

def get_logger(LOG_FILE):

    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    OP_DIR = os.path.join('Logs')

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    handler = logging.FileHandler(os.path.join(OP_DIR, LOG_FILE))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Log start :: ' + str(datetime.now()))
    return logger

def log_time(logger):
    logger.info(str(datetime.now()) + '| Time stamp ' + str(time()))