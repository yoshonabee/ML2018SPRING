import sys
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

# x_train = []
# y_train = []
# f = open(sys.argv[1], 'r')
# row = csv.reader(f, delimiter=' ')
# n_row = 0
# for r in row:
# 	if n_row != 0:
# 		temp = []
# 		for i in range(len(r)):
# 			if i == 0:
# 				y_train.append(int(r[i][0]))
# 				temp.append(int(r[i][2:]))
# 			else:
# 				temp.append(int(r[i]))
# 		x_train.append(temp)
# 	n_row += 1
# f.close()

# np.save("x_train.npy", x_train)
# np.save("y_train.npy", y_train)
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
	