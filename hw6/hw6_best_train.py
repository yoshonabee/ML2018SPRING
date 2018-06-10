import keras
import random
from hw6_utils import *
from keras.optimizers import Adam
from keras.layers import Input

LR = 0.001
EPOCH = 10
BATCH_SIZE = 128
n_vec = 128

users, movies, ratings = load_train_data('train.csv')

data_train = np.concatenate((users, movies, ratings), axis=1).tolist()

user_input = Input(shape=[1])
movie_input = Input(shape=[1])

models = []
for bag in range(10):
	data = [random.choice(data_train) for _ in data_train]
	data = np.array(data)

	u_train = data[:, 0]
	m_train = data[:, 1]
	r_train = data[:, 2]

	n_users = int(np.amax(users))
	n_movies = int(np.amax(movies))

	model = CC(user_input, movie_input, n_users, n_movies, n_vec)
	print(model.summary())

	adam = Adam(LR)
	model.compile(adam, loss="mse")
	model.fit([u_train, m_train], r_train, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, shuffle=True)
	models.append(model)

BigModel = ensemble(user_input, movie_input, models)
BigModel.save('bag.h5')