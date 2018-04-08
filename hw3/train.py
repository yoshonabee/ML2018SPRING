import sys
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

def loadData(filename, mode):
	x_train = []
	y_train = []
	f = open(filename, 'r')
	row = csv.reader(f, delimiter=' ')
	n_row = 0
	for r in row:
		if n_row != 0:
			temp = []
			for i in range(len(r)):
				if i == 0 and mode == 'train':
					y_train.append(int(r[i][0]))
					temp.append(int(r[i][2:]))
				else:
					temp.append(int(r[i]))
			x_train.append(temp)
		n_row += 1
	f.close()

	x_train = np.array(x_train)
	y_train = np.array(y_train)
	x_train.reshape(x_train.shape[0], 48, 48)
	return x_train, y_train

x_train, y_train = loadData(sys.argv[1], 'train')

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
# x_train = np.load("x_train.npy")
# y_train = np.load("y_train.npy")


model = Sequential()
model.add(Conv2D(100, 5, 5, input_shape(1, 48, 48), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(500, 5, 5))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(200, 5, 5))
model.add(Conv2D(2, 2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(7, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x_train, y_train, batch_size=1000, epochs=20)
result = model.evaluate(x_train, y_train, batch_size=1000000)
print('\nTest Accuracy:', result[1])