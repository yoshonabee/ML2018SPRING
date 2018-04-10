import os
import sys
import numpy as np
import tensorflow as tf
from keras.utils import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from hw3_fc import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print("Loading test data...")
x_test = loadData(sys.argv[1], 'test')
model = load_model(sys.argv[2])

print("Predicting...")
y_test = model.predict_classes(x_test)
outputcsv(y_test, sys.argv[3])
print("Done!")

