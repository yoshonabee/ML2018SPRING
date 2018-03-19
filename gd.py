import numpy as np

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

tempx = inputfiles(input(), 3)
tempxt = inputfiles(input(), 2)

##################################################################################################################	

x = []
y = []
tx = [1.0]
for i in range(len(tempx)):
	if (i + 1) % 3 == 0:
		if (tx[2] > 200 or tx[6] > 200 or tempx[i][9] > 200):
			tx = [1.0]
			continue
		y.append(tempx[i][9])
		x.append(tx)
		tx = [1.0]
		continue
	tx += tempx[i][8:11] + [tempx[i][12]]

mx = np.matrix(x)
my = np.matrix(y).T

w = [1.0] * len(x[0])
mw = np.matrix(w).T

xtval = []
ytval = []
tx = [1.0]
for i in range(len(tempxt)):
	if (i + 1) % 3 == 0:
		ytval.append(tempxt[i][9])
		xtval.append(tx)
		tx = [1.0]
		continue
	tx += tempxt[i][8:11] + [tempxt[i][12]]

mxtval = np.matrix(xtval)
mytval = np.matrix(ytval).T

eta = 1
sigma = 0
err = []
for step in range(12000):
	sigma += square(((mx.T * mx * mw - mx.T * my) * 2 / len(x)).tolist())
	mw -= eta / (step + 1) ** (1/2) * (mx.T * mx * mw - mx.T * my) / (sigma / (step + 1)) ** (1/2)
	etout = 0
	# for i in range(len(xtval)):
	# 	etout += (((np.dot(mxtval[i], mw) - mytval[i]).tolist()[0][0] ** 2) / len(xtval))
	# err.append(etout ** (1/2))
##################################################################################################################
# minerr = 100
# for i in range(len(err)):
# 	if minerr > err[i]:
# 		n = i
# 		minerr = err[i]
# print(n, minerr)
ein = 0
for i in range(len(x)):
	ein += (((np.dot(mx[i], mw) - my[i]).tolist()[0][0] ** 2) / len(x))
print('ein =', ein ** (1/2))

for i in range(len(xtval)):
	etout += (((np.dot(mxtval[i], mw) - mytval[i]).tolist()[0][0] ** 2) / len(xtval))
print('etout =', etout ** (1/2))

print(mw)
##################################################################################################################

xtest = []
ytest = []
tx = [1.0]
for i in range(len(tempxt)):
	if (i + 1) % 9 >= 8 or (i + 1) % 9 == 0:
		tx += tempxt[i][8:11] + [tempxt[i][12]]
		if (i + 1) % 9 == 0:
			xtest.append(tx)
			tx = [1.0]
mxtest = np.matrix(xtest)


for i in range(len(xtest)):
	ytest.append(np.dot(mxtest[i], mw).tolist()[0][0])

f = open('out.csv', 'w')
lines = ['id,value\n']
for i in range(len(ytest)):
	lines.append('id_' + str(i) + ',' + str(ytest[i]) + '\n')
f.writelines(lines)
f.close()