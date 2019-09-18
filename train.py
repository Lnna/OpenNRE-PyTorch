from __future__ import division,unicode_literals,print_function

import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_CACHE_PATH']='/home/nana/cudacache'
# CUDA_CACHE_PATH='/home/lnn/cudacache'
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'pcnn_att', help = 'name of the model')
args = parser.parse_args()
model = {
	'pcnn_att': models.PCNN_ATT,
	'pcnn_one': models.PCNN_ONE,
	'pcnn_ave': models.PCNN_AVE,
	'cnn_att': models.CNN_ATT,
	'cnn_one': models.CNN_ONE,
	'cnn_ave': models.CNN_AVE
}
con = config.Config()
<<<<<<< HEAD
con.set_max_epoch(20)
=======
con.set_max_epoch(15)
>>>>>>> bert
con.load_train_data()
con.load_test_data()
con.set_train_model(model[args.model_name])
con.train()
