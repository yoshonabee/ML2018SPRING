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

save = np.load('model.npy')
x = save[0].tolist()
y = save[1].tolist()

tempxt = inputfiles(sys.argv[1], 2)
xtval = []
ytval = []
tx = [1.0, 1.0]
g = 16
for i in range(len(tempxt)):
	if tempxt[i][12] > g: tempxt[i][12] = 0
	if tempxt[i][12] < 0: tempxt[i][12] /= 16
	if tempxt[i][9] < 0: tempxt[i][9] *= -8.1
	tempxt[i][12] /= 10 / 100
	tempxt[i][9] /= 148.0 / 100
	tempxt[i][8] /= 10800.0 / 100
	if (i + 1) % 3 == 0:
		ytval.append(tempxt[i][9])
		xtval.append(tx)
		tx = [1.0, 1.0]
		continue
	tx += [tempxt[i][9], tempxt[i][9] ** 2, tempxt[i][8], tempxt[i][8] ** 2, tempxt[i][12]]



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

xtest = []
ytest = []
tx = [1.0, 1.0]
for i in range(len(tempxt)):
	if (i + 1) % 9 >= 8 or (i + 1) % 9 == 0:
		tx += [tempxt[i][9], tempxt[i][9] ** 2, tempxt[i][8], tempxt[i][8] ** 2, tempxt[i][12]]
		if (i + 1) % 9 == 0:
			xtest.append(tx)
			tx = [1.0, 1.0]
mxtest = np.matrix(xtest)


for i in range(len(xtest)):
	ytest.append(np.dot(mxtest[i], mw).tolist()[0][0] * 148 / 100)

f = open(sys.argv[2], 'w')
lines = ['id,value\n']
for i in range(len(ytest)):
	lines.append('id_' + str(i) + ',' + str(ytest[i]) + '\n')
f.writelines(lines)
f.close()