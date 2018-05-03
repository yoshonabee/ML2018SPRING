import csv
import pickle
import numpy as np

def loadData(filename):
	test_case = []
	f = open(filename, 'r')
	row = csv.reader(f, delimiter=',')
	n_row = 0
	for r in row:
		if n_row != 0:
			temp = []
			for i in r[1:]:
				temp.append(int(i))
			test_case.append(temp)
		n_row += 1
	f.close()

	return np.array(test_case)

def outputcsv(y, filename):
	f = open(filename, 'w')
	lines = ['ID,Ans\n']

	for i in range(len(y)):
		lines.append(str(i) + ',' + str(y[i]) + '\n')
	f.writelines(lines)
	f.close()

def save_model(model, filename):
	with open(filename, 'wb') as f:
		pickle.dump(model, f)

def load_model(filename):
	with open(filename, 'rb') as f:
		model = pickle.load(f)

	return model

def nor(x):
	x = (x - np.mean(x)) / np.var(x)
	return x