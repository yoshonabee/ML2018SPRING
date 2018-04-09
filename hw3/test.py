import os
import sys
import numpy as np
import tensorflow as tf
from keras.utils import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from hw3_fc import loadData

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
set_session(tf.Session(config=config))

model = load_model('model.h5')
x_test = loadData(sys.argv[1], 'test')

y_test = model.predict(x_test)
for i in y_test:
	print(i)
