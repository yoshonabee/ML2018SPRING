import os
import sys
import numpy as np
import tensorflow as tf
from keras.utils import *
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from hw3_fc import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
set_session(tf.Session(config=config))


# x_train, y_train = loadData(sys.argv[1], 'train')
# np.save("x_train.npy", x_train)
# np.save("y_train.npy", y_train)
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1363)
gen = ImageDataGenerator(featurewise_center=False,
						samplewise_center=False,
						rotation_range=30,
						width_shift_range=0.2,
						shear_range=0.2,
						height_shift_range=0.2,
						zoom_range=0.2,
						data_format='channels_last')

gen.fit(x_train)
train_generator = gen.flow(x_train, y_train, batch_size=100)

dropout_rate = 0
model = Sequential()
model.add(Conv2D(60, (5, 5), input_shape= (48, 48, 1), data_format='channels_last', activation='relu'))
model.add(Dropout(dropout_rate))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(120, (3, 3), data_format='channels_last', activation='relu'))
model.add(Dropout(dropout_rate))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(240, (3, 3), data_format='channels_last', activation='relu'))

# model.add(Conv2D(200, (5, 5), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(dropout_rate))
model.add(Dense(1363, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(7, activation='softmax'))
print(model.summary())

if (input("\nRun? Y/n: ") == 'y'):
	epoch = int(input("epoch = "))
	adam = Adam(lr=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	train_history = model.fit_generator(train_generator, steps_per_epoch=600, validation_data=(x_val, y_val), epochs=epoch)
	model.save(sys.argv[2])
# result = model.evaluate(x_train, y_train, batch_size=100000)
# print('\nTest Accuracy:', result[1])