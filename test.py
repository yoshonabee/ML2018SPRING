import numpy as np
import funcs as fc

tempx = fc.inputfiles('train.csv', 3)
tempxt = fc.inputfiles(input(), 2)

##################################################################################################################	


for bound in range(2, 10):
	x = []
	y = []
	xval = []
	yval = []
	tx = [1.0]
	k = 1
	count = 0
	print(bound)
	for i in range(len(tempx)):
		if (tempx[i][9] > 200):
			count = 0
			tx = [1.0]
			i += (3 - i % 3) % 3
			continue
		if (count + 1) % 3 == 0:
			if k % bound == 0:
				xval.append(tx)
				yval.append(tempx[i][9])
			else:
				y.append(tempx[i][9])
				x.append(tx)
			tx = [1.0]
			k += 1
			count += 1
			continue
		tx += tempx[i][8:11] + [tempx[i][12]]
		count += 1

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

	x += xtval * 2
	y += ytval * 2
	mx = np.matrix(x)
	my = np.matrix(y).T
	mxval = np.matrix(xval)
	myval = np.matrix(yval).T
	mxtval = np.matrix(xtval)
	mytval = np.matrix(ytval).T
	w = np.matrix([1.0] * len(x[0])).T

	##################################################################################################################
	w = (mx.T * mx).I * mx.T * my
	##################################################################################################################

	ein = 0
	for i in range(len(x)):
		ein += (((np.dot(mx[i], w) - my[i]).tolist()[0][0] ** 2) / len(x))
	print('ein =', ein ** (1/2))

	# takeout = []
	# for i in range(len(xval)):
	# 	eout = (((np.dot(mxval[i], w) - myval[i]).tolist()[0][0] ** 2) ** (1/2))
	# 	if eout > 30:
	# 		takeout.append(i)
	# print(takeout)

	etout = 0
	for i in range(len(xtval)):
		etout += (((np.dot(mxtval[i], w) - mytval[i]).tolist()[0][0] ** 2) / len(xtval))
	print('etout =', etout ** (1/2))

	print(w)
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
	ytest.append(np.dot(mxtest[i], w).tolist()[0][0])

fc.outputfiles(input(), ytest)