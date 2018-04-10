import csv
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt

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

def save_train_result(train_history, filename):
	plt.plot(train_history.history['acc'])
	plt.plot(train_history.history['val_acc'])
	plt.title('Train History')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(filename)