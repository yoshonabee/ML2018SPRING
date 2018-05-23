import csv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

LR = 0.001
EPOCH = 20
VEC_SIZE = 200
BATCH_SIZE = 256
SENTENCE_LENGTH = 40

class RNN01(nn.Module):
	def __init__(self):
		super(RNN01, self).__init__()
		self.gru1 = nn.GRU(VEC_SIZE, 200, bidirectional=True, batch_first=True, dropout=0.3)
		self.gru2 = nn.GRU(200 * 2, 16, bidirectional=True, batch_first=True, dropout=0.3)
		self.l1 = nn.Linear(SENTENCE_LENGTH * 16 * 2, SENTENCE_LENGTH * 16)
		self.l2 = nn.Linear(SENTENCE_LENGTH * 16, SENTENCE_LENGTH * 8)
		self.l3 = nn.Linear(SENTENCE_LENGTH * 8, SENTENCE_LENGTH * 4)
		self.out = nn.Linear(SENTENCE_LENGTH * 4, 2)

	def forward(self, x):
		self.gru1.flatten_parameters()
		x, _ = self.gru1(x)
		self.gru2.flatten_parameters()
		x, _ = self.gru2(x)
		x = x.contiguous().view(x.size(0), -1)
		x = self.l1(x)
		x = x * F.sigmoid(x)
		x = self.l2(x)
		x = x * F.sigmoid(x)
		x = self.l3(x)
		x = x * F.sigmoid(x)
		out = self.out(x)
		return out

class RNN02(nn.Module):
	def __init__(self):
		super(RNN02, self).__init__()
		self.gru1 = nn.GRU(VEC_SIZE, 512, bidirectional=True, batch_first=True, dropout=0.3)
		self.gru2 = nn.GRU(512 * 2, 8, bidirectional=True, batch_first=True, dropout=0.3)
		self.l1 = nn.Linear(SENTENCE_LENGTH * 8 * 2, SENTENCE_LENGTH * 8)
		self.l2 = nn.Linear(SENTENCE_LENGTH * 8, SENTENCE_LENGTH * 4)
		self.l3 = nn.Linear(SENTENCE_LENGTH * 4, SENTENCE_LENGTH * 2)
		self.out = nn.Linear(SENTENCE_LENGTH * 2, 2)

	def forward(self, x):
		self.gru1.flatten_parameters()
		x, _ = self.gru1(x)
		self.gru2.flatten_parameters()
		x, _ = self.gru2(x)
		x = x.contiguous().view(x.size(0), -1)
		x = self.l1(x)
		x = x * F.sigmoid(x)
		x = self.l2(x)
		x = x * F.sigmoid(x)
		x = self.l3(x)
		x = x * F.sigmoid(x)
		out = self.out(x)
		return out

class RNN03(nn.Module):
	def __init__(self):
		super(RNN03, self).__init__()
		self.gru1 = nn.GRU(VEC_SIZE, 128, bidirectional=True, batch_first=True, dropout=0.3)
		self.gru2 = nn.GRU(128 * 2, 64, bidirectional=True, batch_first=True, dropout=0.3)
		self.l1 = nn.Linear(SENTENCE_LENGTH * 64 * 2, SENTENCE_LENGTH * 64)
		self.l2 = nn.Linear(SENTENCE_LENGTH * 64, SENTENCE_LENGTH * 32)
		self.l3 = nn.Linear(SENTENCE_LENGTH * 32, SENTENCE_LENGTH * 16)
		self.out = nn.Linear(SENTENCE_LENGTH * 16, 2)

	def forward(self, x):
		self.gru1.flatten_parameters()
		x, _ = self.gru1(x)
		self.gru2.flatten_parameters()
		x, _ = self.gru2(x)
		x = x.contiguous().view(x.size(0), -1)
		x = self.l1(x)
		x = x * F.sigmoid(x)
		x = self.l2(x)
		x = x * F.sigmoid(x)
		x = self.l3(x)
		x = x * F.sigmoid(x)
		out = self.out(x)
		return out

class Ensemble(nn.Module):
	def __init__(self, rnn01, rnn02, rnn03):
		super(Ensemble, self).__init__()
		self.rnn01 = rnn01
		self.rnn02 = rnn02
		self.rnn03 = rnn03

	def forward(self, x):
		x1 = F.softmax(self.rnn01(x), dim=1)
		x2 = F.softmax(self.rnn02(x), dim=1)
		x3 = F.softmax(self.rnn03(x), dim=1)
		return x1 + x2 + x3
		
def loadData(filename, sentence_length=10, label=True, train=True):
	f = open(filename, 'r', encoding='utf-8')
	
	if train:
		if label:
			x, y = [], []
			for sentence in f:
				string = sentence[10:]
				string = string.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt")\
								.replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")\
								.replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")\
								.replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")\
								.replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")\
								.replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")\
								.replace("couldn ' t","couldnt")
				x.append(string.split()[:sentence_length])
				y.append(sentence[0])
			f.close()

			

			y = np.array(y).astype(np.int64)

			return x, y
		else:
			x = []
			for sentence in f:
				sentence = sentence.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt")\
								.replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")\
								.replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")\
								.replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")\
								.replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")\
								.replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")\
								.replace("couldn ' t","couldnt")
				x.append(sentence.split()[:sentence_length])
			f.close()

			return x
	else:
		x = []
		n_row = 0
		for sentence in f:
			if n_row != 0:
				string = sentence[len(str(n_row - 1)) + 1:]
				string = string.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt")\
								.replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")\
								.replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")\
								.replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")\
								.replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")\
								.replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")\
								.replace("couldn ' t","couldnt")
				x.append(string.split()[:sentence_length])
			n_row += 1
		f.close()

		return x

class Data(Dataset):
	def __init__(self, x, y=None, word_vector=None):
		self.x = np.array(x)
		if y is not None: self.y = np.array(y)
		else: self.y = None
		self.wv = word_vector

	def __getitem__(self, index):
		if self.y is not None:
			return self.transform(self.x[index]), self.y[index]
		else:
			return self.transform(self.x[index])

	def __len__(self):
		return self.x.shape[0]

	def transform(self, x):
		fill = [0] * 200
		result = []

		for i in x:
			if i not in self.wv:
				result.append(fill)
			else:
				result.append(self.wv[i])

		while len(result) < 40:
			result.append(fill)
		return torch.from_numpy(np.array(result).astype(np.float32))

def outputcsv(y, filename):
	f = open(filename, 'w')
	lines = ['id,label\n']

	for i in range(len(y)):
		lines.append(str(i) + ',' + str(y[i]) + '\n')
	f.writelines(lines)
	f.close()