import os
import sys
import csv
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
set_session(tf.Session(config=config))

def loadData(filename, mode):
	x_train = []
	x_val = []
	y_val = []
	y_temp = []
	f = open(filename, 'r')
	row = csv.reader(f, delimiter=' ')
	n_row = 0
	for r in row:
		if n_row != 0:
			temp = []
			for i in range(len(r)):
				if i == 0 and mode == 'train':
					y_temp.append(int(r[i][0]))
					temp.append(int(r[i][2:]))
				else:
					temp.append(int(r[i]))
			x_train.append(temp)
		n_row += 1
	f.close()

	y_train = [[0] * 7] * len(y_temp)
	for i in range(len(y_temp)):
		y_train[i][y_temp[i]] = 1

	x_train = np.array(x_train)
	y_train = np.array(y_train)
	x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
	return x_train, y_train

x_train, y_train = loadData(sys.argv[1], 'train')
print(x_train.shape)
print(y_train.shape)
np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
# x_train = np.load("x_train.npy")
# y_train = np.load("y_train.npy")


model = Sequential()
model.add(Conv2D(100, (5, 5), input_shape= (48, 48, 1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(100, (5, 5)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(100, (5, 5)))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(2000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(7, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x_train, y_train, validation_split=0.2, batch_size=1000, epochs=5)
# result = model.evaluate(x_train, y_train, batch_size=100000)
# print('\nTest Accuracy:', result[1])