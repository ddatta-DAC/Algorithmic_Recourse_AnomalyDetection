import os
import sys
import pandas as pd
import argparse
import numpy as np
import AD_Explainer

# ------------------------------------------------- # 
if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Pretrain the encoder through MLM like objective')
    parser.add_argument(
        '--dir', 
        type=str,
        choices = ['us_import1', 'us_import2', 'us_import3','ecuador_export','colombia_export'],
        help='Which data subset to use ?'
    )

    args = parser.parse_args()
    DIR = args.dir
    AD_Explainer.train_model(subDIR=DIR)
    