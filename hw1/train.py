import numpy as np
import sys

def square(grad):
	result = 0
	for i in grad:
		result += i[0] ** 2
	return result

def inputfiles(filename, n):
	result = []
	f = open(filename, 'r', encoding='Big5')
	data = []
	lines = f.readlines();
	f.close()

	for i in range(n - 2, len(lines)):
		temp = []
		s = ''
		for j in range(len(lines[i]) - 1):
			if lines[i][j] != ',':
				s += lines[i][j]
			else:
				if s == 'NR':
					s = '0'
				temp.append(s)
				s = ''
		
		if s == 'NR':
			s = '0'
		temp.append(s)
		data.append(temp)

	for i in range(len(data)):
		data[i] = data[i][n:]

	for i in range(18, len(data)):
		data[i % 18] += data[i]

	data = data[0:18]

	for i in range(len(data[0])):
		tx = []
		for j in range(len(data)):
			tx.append(float(data[j][i]))
		result.append(tx)

	return result

tempx = inputfiles("train.csv", 3)
##################################################################################################################	
x = []
y = []
tx = [1.0, 1.0]
for i in range(len(tempx)):
	if tempx[i][12] > 16: tempx[i][12] = 0
	if tempx[i][12] < 0: tempx[i][12] /= 16
	if tempx[i][9] < 0: tempx[i][9] *= -8.1
	if tempx[i][9] > 200: tempx[i][9] /= 8.2
	tempx[i][9] /= 148.0 / 100
	tempx[i][8] /= 10800.0 / 100
	tempx[i][12] /= 10 / 100
	if (i + 1) % 3 == 0:
		y.append(tempx[i][9])
		x.append(tx)
		tx = [1.0, 1.0]
		continue
	tx += [tempx[i][9], tempx[i][9] ** 2, tempx[i][8], tempx[i][8] ** 2, tempx[i][12]]

load = np.load('model.npy')
xtval = load[0]
ytval = load[1]
save = [xtval, ytval]

x += xtval * 3
y += ytval * 3
mx = np.matrix(x)
my = np.matrix(y).T
mxtval = np.matrix(xtval)
mytval = np.matrix(ytval).T
w = [1.0] * len(x[0])
mw = np.matrix(w).T

eta = 1
sigma = 0
Lambda = 100000
for step in range(28000):
	sigma += square(((mx.T * (np.dot(mx, mw) - my) + Lambda * mw) / len(x)).tolist())
	mw -= eta * (mx.T * (np.dot(mx, mw) - my) + Lambda * mw) / sigma ** (1/2)
save.append(mw)
np.save('model.npy', save)