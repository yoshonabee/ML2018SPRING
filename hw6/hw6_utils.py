import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Dense, Flatten, Dropout, merge, Embedding, Add, Concatenate, Average
from keras.models import Model, load_model
from keras.initializers import glorot_normal

def load_train_data(filename):
	data_df = pd.read_csv(filename, sep=',')

	users, movies, ratings= [], [], []
	for i in range(data_df.shape[0]):
		if (i + 1) % 10000 == 0:
			print(i + 1)
		users.append(int(data_df['UserID'][i]))
		movies.append(int(data_df['MovieID'][i]))
		ratings.append(int(data_df['Rating'][i]))

	return np.array(users), np.array(movies), np.array(ratings)

def load_test_data(filename):
	case_df = pd.read_csv(filename)
	
	users, movies = [], []
	for i in range(case_df.shape[0]):
		users.append(int(case_df['UserID'][i]))
		movies.append(int(case_df['MovieID'][i]))

	return np.array(users).reshape(-1, 1), np.array(movies).reshape(-1, 1)

def outputcsv(y, filename):
	f = open(filename, 'w')
	lines = ['TestDataID,Rating\n']

	for i in range(len(y)):
		lines.append(str(i + 1) + ',' + str(y[i]) + '\n')
	f.writelines(lines)
	f.close()

def MF(u, m, n_user, n_movie, n_vec):
	user = Embedding(n_user + 1, n_vec, embeddings_initializer=glorot_normal())(u)
	user_b = Embedding(n_user + 1, 1, embeddings_initializer=glorot_normal())(u)
	movie = Embedding(n_movie + 1, n_vec, embeddings_initializer=glorot_normal())(m)
	movie_b = Embedding(n_movie + 1, 1, embeddings_initializer=glorot_normal())(m)

	user = Flatten()(user)
	user_b = Flatten()(user_b)
	movie = Flatten()(movie)
	movie_b = Flatten()(movie_b)

	out = merge([user, movie], mode='dot')
	out = Add()([out, user_b, movie_b])

	model = Model(input=[u, m], output=out)
	return model

def CC(u, m, n_user, n_movie, n_vec):
	user = Embedding(n_user + 1, n_vec, embeddings_initializer=glorot_normal())(u)
	user_b = Embedding(n_user + 1, 1, embeddings_initializer=glorot_normal())(u)
	movie = Embedding(n_movie + 1, n_vec, embeddings_initializer=glorot_normal())(m)
	movie_b = Embedding(n_movie + 1, 1, embeddings_initializer=glorot_normal())(m)

	user = Flatten()(user)
	user_b = Flatten()(user_b)
	movie = Flatten()(movie)
	movie_b = Flatten()(movie_b)


	out = merge([user, movie], mode='concat')
	out = Dense(256, activation='relu')(out)
	out = Dropout(0.2)(out)
	out = Dense(128, activation='relu')(out)
	out = Dropout(0.2)(out)
	out = Dense(64, activation='relu')(out)
	out = Dropout(0.2)(out)
	out = Dense(32, activation='relu')(out)
	out = Dropout(0.2)(out)
	out = Dense(16, activation='relu')(out)
	out = Dropout(0.2)(out)
	out = Dense(1)(out)
	out = Add()([out, user_b, movie_b])

	model = Model(input=[u, m], output=out)
	return model

def ensemble(u, m, models):
	outputs = [model.outputs[0] for model in models]
	y = Average()(outputs)
	model = Model(inputs=[u, m], outputs=y)
	return model