###! /usr/bin/env python
### -*- coding:utf-8 -*-

import os, sys
import time
import numpy
import operator
import pandas as pd
#import torch
#import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import *
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.inspection import permutation_importance

model_directory = "./../"

#===============================================================
# LOAD THE MODEL USING JOBLIB
#===============================================================
import joblib
filename = model_directory+'best/rf_model.joblib'

model = joblib.load(filename)
print("Model loaded")

#===============================================================
# COMPUTE OUTPUT
#===============================================================
inputs = numpy.loadtxt("./features.txt")
print(inputs[0])

pred_outputs = model.predict(inputs)
print(pred_outputs[0])

numpy.savetxt("field_param.txt",pred_outputs)
