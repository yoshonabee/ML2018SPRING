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

def outputfiles(filename, ytest):
	f = open(filename, 'w')
	lines = ['id,value\n']
	for i in range(len(ytest)):
		lines.append('id_' + str(i) + ',' + str(ytest[i]) + '\n')
	f.writelines(lines)
	f.close()