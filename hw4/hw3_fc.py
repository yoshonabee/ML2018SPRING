import csv
import math
import numpy as np
from keras.models import Model, Input
from keras.utils import np_utils
from keras.layers import *
from keras.layers.merge import *
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import itertools


def loadData(filename, mode):
	x = []
	y = []
	f = open(filename, 'r')
	row = csv.reader(f, delimiter=' ')
	n_row = 0
	for r in row:
		if n_row != 0:
			temp = []
			for i in range(len(r)):
				if i == 0:
					if mode == 'train': 
						y.append(int(r[0][0]))
						temp.append(int(r[0][2:]))
					else:
						temp.append(int(r[0][len(str(n_row)) + 1:]))
				else:
					temp.append(int(r[i]))
			x.append(temp)
		n_row += 1
	f.close()

	x = np.array(x)
	y = np.array(y)
	y = np_utils.to_categorical(y, num_classes=7)
	x = x.reshape(x.shape[0], 48, 48, 1)
	x = x / 255

	if mode == 'train': return x, y
	else: return x

def outputcsv(y, filename):
	f = open(filename, 'w')
	lines = ['id,label\n']

	for i in range(len(y)):
		lines.append(str(i) + ',' + str(y[i]) + '\n')
	f.writelines(lines)
	f.close()

def stack(x, h):
	conv = Conv2D(h // 4, (1, 1), activation='relu', padding='same')(x)
	conv = Conv2D(h // 4, (3, 3), activation='relu', padding='same')(conv)
	conv = Conv2D(h, (1, 1), activation='relu', padding='same')(conv)
	result = add([x, conv])
	return Activation('relu')(result)

def data_normalization(x):
	for i in range(len(x)):
		if i % 5000 == 0: print("Normalizing feature %d" %(i))
		mean = 0.0

		for feature in range(len(x[i])):
			mean += x[i][feature]
		mean /= len(x[i])
		sd = 0
		for feature in range(len(x[i])):
			sd += (x[i][feature] - mean) ** 2 / len(x[i])
		sd = math.sqrt(sd)
		if sd == 0:
			for feature in range(len(x[i])): x[i][feature] = 0
		else:
			for feature in range(len(x[i])): x[i][feature] = (x[i][feature] - mean) / sd
	print("----Done----")
	return x

def genImage(x, y):
	gen = ImageDataGenerator(	featurewise_center=False,
								samplewise_center=False,
								rotation_range=15,
								width_shift_range=0.3,
								shear_range=0.3,
								height_shift_range=0.3,
								zoom_range=0.3,
								data_format='channels_last')

	gen.fit(x, augment=True, rounds=5)
	return gen.flow(x, y, batch_size=512, seed=120)

def ensemble(inputs, models):
	outputs = [model.outputs[0] for model in models]
	y = Average()(outputs)
	model = Model(inputs=inputs, outputs=y)
	return model

def gen08(inputs):
	dropout_rate = 0.2
	model = Conv2D(60, (5, 5), input_shape= (48, 48, 1), activation='relu')(inputs)
	model = Dropout(dropout_rate)(model)
	model = MaxPooling2D((2, 2))(model)
	model = Conv2D(120, (3, 3), activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = MaxPooling2D((2, 2))(model)
	model = Conv2D(240, (3, 3), activation='relu')(model)
	model = MaxPooling2D((2, 2))(model)

	model = Flatten()(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(400, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(200, activation='relu')(model)
	model = Dense(7, activation='softmax')(model)
	model = Model(inputs=inputs, outputs=model)
	return model

def gen09(inputs):
	dropout_rate = 0.2
	model = Conv2D(60, (5, 5), input_shape= (48, 48, 1), activation='relu')(inputs)
	model = Dropout(dropout_rate)(model)
	model = MaxPooling2D((2, 2))(model)
	model = Conv2D(120, (3, 3), activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = MaxPooling2D((2, 2))(model)
	model = Conv2D(240, (3, 3), activation='relu')(model)
	model = MaxPooling2D((2, 2))(model)

	model = Flatten()(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(400, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(200, activation='relu')(model)
	model = Dense(100, activation='relu')(model)
	model = Dense(7, activation='softmax')(model)
	model = Model(inputs=inputs, outputs=model)
	return model

def gen12(inputs):
	dropout_rate = 0.2
	model = Conv2D(60, (5, 5), input_shape= (48, 48, 1), activation='relu')(inputs)
	model = Dropout(dropout_rate)(model)
	model = MaxPooling2D((2, 2))(model)
	model = Conv2D(120, (3, 3), activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = MaxPooling2D((2, 2))(model)
	model = Conv2D(240, (3, 3), activation='relu')(model)
	model = MaxPooling2D((2, 2))(model)

	model = Flatten()(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(300, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(300, activation='relu')(model)
	model = Dense(300, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(300, activation='relu')(model)
	model = Dense(300, activation='relu')(model)
	model = Dense(7, activation='softmax')(model)
	model = Model(inputs=inputs, outputs=model)
	return model

def func01(inputs):
	dropout_rate = 0.2

	model1 = Conv2D(60, (3, 3), activation='relu')(inputs)
	model1 = Conv2D(60, (3, 3), activation='relu')(model1)
	model1 = MaxPooling2D(2, 2)(model1)
	model1 = Conv2D(120, (3, 3), activation='relu')(model1)
	model1 = MaxPooling2D(2, 2)(model1)
	model1 = Conv2D(240, (3, 3), activation='relu')(model1)
	model1 = Flatten()(model1)

	model2 = MaxPooling2D(2, 2)(inputs)
	model2 = Conv2D(60, (3, 3), activation='relu')(model2)
	model2 = MaxPooling2D(2, 2)(model2)
	model2 = Conv2D(120, (3, 3), activation='relu')(model2)
	model2 = MaxPooling2D(2, 2)(model2)
	model2 = Flatten()(model2)

	model = concatenate([model1, model2])

	model = Dense(300, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(300, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(300, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(300, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(300, activation='relu')(model)
	model = Dense(7, activation='softmax')(model)

	model = Model(inputs=inputs, outputs=model)
	return model

def func02(inputs):
	dropout_rate = 0.25

	model1 = Conv2D(60, (3, 3), activation='relu')(inputs)
	model1 = Conv2D(60, (3, 3), activation='relu')(model1)
	model1 = MaxPooling2D(2, 2)(model1)
	model1 = Conv2D(120, (3, 3), activation='relu')(model1)
	model1 = MaxPooling2D(2, 2)(model1)
	model1 = Conv2D(240, (3, 3), activation='relu')(model1)
	model1 = MaxPooling2D(2, 2)(model1)
	model1 = Flatten()(model1)

	model2 = MaxPooling2D(2, 2)(inputs)
	model2 = Conv2D(80, (3, 3), activation='relu')(model2)
	model2 = MaxPooling2D(2, 2)(model2)
	model2 = Conv2D(160, (3, 3), activation='relu')(model2)
	model2 = MaxPooling2D(2, 2)(model2)
	model2 = Flatten()(model2)

	model = concatenate([model1, model2])

	model = Dense(300, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(300, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(300, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(300, activation='relu')(model)
	model = Dropout(dropout_rate)(model)
	model = Dense(300, activation='relu')(model)
	model = Dense(7, activation='softmax')(model)

	model = Model(inputs=inputs, outputs=model)
	return model

def Res02(inputs):
	dropout_rate = 0.15

	model = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
	model = stack(model, 64)
	model = MaxPooling2D((2, 2), padding='same')(model)
	model = Dropout(dropout_rate)(model)
	model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
	model = stack(model, 128)
	model = MaxPooling2D((2, 2), padding='same')(model)
	model = Dropout(dropout_rate)(model)
	model = Conv2D(256, (3, 3), activation='relu', padding='same')(model)
	model = stack(model, 256)
	model = MaxPooling2D((2, 2), padding='same')(model)
	model = Dropout(dropout_rate)(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same')(model)
	model = AveragePooling2D((2, 2), padding='same')(model)
	model = Dropout(dropout_rate)(model)

	model = Flatten()(model)
	model = Dense(400, activation='relu')(model)
	model = Dense(7, activation='softmax')(model)

	model = Model(inputs=inputs, outputs=model)
	return model

def Res03(inputs):
	dropout_rate = 0.15

	model = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
	model = stack(model, 64)
	model = MaxPooling2D((2, 2), padding='same')(model)
	model = Dropout(dropout_rate)(model)
	model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
	model = stack(model, 128)
	model = stack(model, 128)
	model = MaxPooling2D((2, 2), padding='same')(model)
	model = Dropout(dropout_rate)(model)
	model = Conv2D(256, (3, 3), activation='relu', padding='same')(model)
	model = stack(model, 256)
	model = stack(model, 256)
	model = stack(model, 256)
	model = MaxPooling2D((2, 2), padding='same')(model)
	model = Dropout(dropout_rate)(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same')(model)
	model = AveragePooling2D((2, 2), padding='same')(model)
	model = Dropout(dropout_rate)(model)

	model = Flatten()(model)
	model = Dense(512, activation='relu')(model)
	model = Dense(256, activation='relu')(model)
	model = Dense(128, activation='relu')(model)
	model = Dense(7, activation='softmax')(model)

	model = Model(inputs=inputs, outputs=model)
	return model

def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print(cm)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], '.2f'),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
