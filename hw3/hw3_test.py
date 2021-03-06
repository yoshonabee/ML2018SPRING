import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from hw3_fc import *

print("\n======================================================Loading Testing Data======================================================\n")
x_test = loadData(sys.argv[1], 'test')

model = load_model('model.h5')

print("\n===========================================================Predicting===========================================================\n")
y_test = model.predict(x_test)

result = []
for i in range(len(y_test)):
	max = 0
	for j in range(len(y_test[0])):
		if max < y_test[i][j]:
			max = y_test[i][j]
			index = j
	result.append(index)
outputcsv(result, sys.argv[2])
print("\n==============================================================Done==============================================================\n")

