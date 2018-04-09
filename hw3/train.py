import os
import sys
import numpy as np
import tensorflow as tf
from keras.utils import *
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from hw3_fc import loadData

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
set_session(tf.Session(config=config))


x_train, y_train = loadData(sys.argv[1], 'train')
np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
# x_train = np.load("x_train.npy")
# y_train = np.load("y_train.npy")



model = Sequential()
model.add(Conv2D(48, (5, 5), input_shape= (48, 48, 1), data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(96, (5, 5), data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (5, 5), data_format='channels_last'))
model.add(Activation('relu'))

# model.add(Conv2D(200, (5, 5), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(7, activation='softmax'))
print(model.summary())

if (input("\nRun? Y/n: ") == 'y'):
	epoch = int(input("epoch = "))
	adam = Adam(lr=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	train_history = model.fit(x_train, y_train, validation_split=0.2, batch_size=500, epochs=epoch)
	model.save('model.h5')
# result = model.evaluate(x_train, y_train, batch_size=100000)
# print('\nTest Accuracy:', result[1])