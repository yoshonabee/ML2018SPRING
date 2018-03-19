import numpy as np
import matplotlib.pyplot as plt

##############################################################################################################
##############################################################################################################

filename = 'test.csv'
f = open(filename, 'r', encoding='Big5')
bdata = []
data = f.readlines();
f.close()
del f

for i in range(0, len(data)):
	temp = []
	s = ''
	for j in range(len(data[i]) - 1):
		if data[i][j] != ',':
			s += data[i][j]
		else:
			if s == 'NR':
				s = '0'
			temp.append(s)
			s = ''
	
	if s == 'NR':
		s = '0'
	temp.append(s)
	bdata.append(temp)

data = []
for i in bdata:
	data.append(i[2:])
del bdata

for j in range(18, len(data)):
	data[j % 18] += data[j]

data = data[0:18]
tempx = []

for i in range(len(data[0])):
	tx = []
	for j in range(len(data)):
		tx.append(float(data[j][i]))
	tempx.append(tx)


x = []
y = []
tx = [1.0]
for i in range(len(tempx)):
	x.append(tempx[i][12])
	y.append(tempx[i][9])

plt.scatter(x, y, s = 1)
plt.show()

	


