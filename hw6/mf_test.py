import os
import sys
import pandas as pd
import numpy as np
import keras
from keras.models import Model, load_model
from hw6_utils import *

u, m = load_test_data(sys.argv[1])
model = load_model('mf.h5')

y = model.predict([u, m]).reshape(-1)

for i in range(len(y)):
	if y[i] > 5:
		y[i] = 5
	elif y[i] < 1:
		y[i] = 1

outputcsv(y, sys.argv[2])
