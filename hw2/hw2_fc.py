import csv
import math

noz = [0, 10, 78, 79, 80]

def theta(z): return 1 / (1 + math.exp(-z))

def sign(x): return 1 if x > 0.5 else 0

def readX(filename):
	x = []
	f = open(filename, 'r') 
	row = csv.reader(f , delimiter=",")
	n_row = 0
	for r in row:
		if n_row != 0:
			temp = []
			for i in range(len(r)):	temp.append(int(r[i]))
			x.append(temp)
		n_row += 1
	f.close()
	return x

def readY(filename):
	y = []
	f = open(filename, 'r')
	ys = f.read()
	for i in ys:
		if i == '0':
			y.append(0)
		elif i == '1':
			y.append(1)
	f.close()
	return y

def norFeature(x):
	print(len(x[0]))
	for feature in range(len(x[0])):
		if feature % 100 == 0: print("Generalize feature %d" %(feature))
		mean = 0.0
		for i in range(len(x)):
			mean += x[i][feature]
		mean /= len(x)
		sd = 0
		for i in range(len(x)):
			sd += (x[i][feature] - mean) ** 2 / len(x)
		sd = math.sqrt(sd)
		if sd == 0:
			for i in range(len(x)): x[i][feature] = 0
		else:
			for i in range(len(x)): x[i][feature] = (x[i][feature] - mean) / sd / 10
	return x

def addFeature(x):
	for i in range(len(x[0])):
		if i % 10 == 0: print("Add feature %d" %(i))
		if i in noz:
			for j in range(len(x)):
				add = x[j][i] / 2
				x[j] += [add ** 2, add ** 3, add ** 4, add ** 5, add ** 6,
						add ** 7, add ** 8, add ** 9, add ** 10, add ** 11,
						add ** 12, add ** 13, add ** 14, add ** 15, add ** 16,
						add ** 17, add ** 18, add ** 19, add ** 20, add ** 21,
						add ** 22, add ** 23, add ** 24, add ** 25, add ** 26,
						add ** 27, add ** 28, add ** 29, add ** 30, add ** 31,
						add ** 32, add ** 33, add ** 34, add ** 35, add ** 36,
						add ** 37, add ** 38, add ** 39, add ** 40, add ** 41,
						add ** 42, add ** 43, add ** 44, add ** 45, add ** 46,
						add ** 47, add ** 48, add ** 49, add ** 50, add ** 51,
						add ** 52, add ** 53, add ** 54, add ** 55, add ** 56,
						add ** 57, add ** 58, add ** 59, add ** 60, add ** 61,
						add ** 62, add ** 63, add ** 64, add ** 65, add ** 66,
						add ** 67, add ** 68, add ** 69, add ** 70, add ** 71,
						add ** 72, add ** 73, add ** 74, add ** 75, add ** 76,
						add ** 77, add ** 78, add ** 79, add ** 80, add ** 81,
						add ** 82, add ** 83, add ** 84, add ** 85, add ** 86,
						add ** 87, add ** 88, add ** 89, add ** 90, add ** 91,
						add ** 92, add ** 93, add ** 94, add ** 95, add ** 96,
						add ** 97, add ** 98, add ** 99, add ** 100, add ** 101,
						add ** 102, add ** 103, add ** 104, add ** 105, add ** 106,
						add ** 107, add ** 108, add ** 109, add ** 110, add ** 111,
						add ** 112, add ** 113, add ** 114, add ** 115, add ** 116,
						add ** 117, add ** 118, add ** 119, add ** 120, add ** 121,
						add ** 122, add ** 123, add ** 124, add ** 125, add ** 126,
						add ** 127, add ** 128, add ** 129, add ** 130, add ** 131,
						add ** 132, add ** 133, add ** 134, add ** 135, add ** 136,
						add ** 137, add ** 138, add ** 139, add ** 140, add ** 141,
						add ** 142, add ** 143, add ** 144, add ** 145, add ** 146,
						add ** 147, add ** 148, add ** 149, add ** 150, add ** 151,
						add ** 152, add ** 153, add ** 154, add ** 155, add ** 156,
						add ** 157, add ** 158, add ** 159, add ** 160, add ** 161,
						add ** 162, add ** 163, add ** 164, add ** 165, add ** 166,
						add ** 167, add ** 168, add ** 169, add ** 170, add ** 171,
						add ** 172, add ** 173, add ** 174, add ** 175, add ** 176,
						add ** 177, add ** 178, add ** 179, add ** 180, add ** 181,
						math.sin(add), math.cos(add), math.tan(add), math.atan(add),]
	return x

def fit(x):
	x = norFeature(x)
	x = addFeature(x)
	x = norFeature(x)
	return x

def outputcsv(y, filename):
	f = open(filename, 'w')
	lines = ['id,label\n']

	for i in range(len(y)):
	    lines.append(str(i + 1) + ',' + str(y[i]) + '\n')
	f.writelines(lines)
	f.close()