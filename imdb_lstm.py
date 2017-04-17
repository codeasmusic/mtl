'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM

import sys
import json
import numpy as np
import mytools.imdb as imdb


def load_data(num_words, skip_top=0, path=sys.path[0]+'/data/imdb.npz'):
	print('Loading data...', num_words)

	input_file = np.load(path)
	x_train = input_file['x_train']	# 2D array, each row is [index of reivew's words]
	y_train = input_file['y_train']
	x_test = input_file['x_test']
	y_test = input_file['y_test']
	input_file.close()

	x_train = [filter(lambda w: w >= skip_top and w < num_words, x) for x in x_train]
	x_test = [filter(lambda w: w >= skip_top and w < num_words, x) for x in x_test]
	
	return (x_train, y_train), (x_test, y_test)


# get dict: {word: embedding}
def get_word_embedding(glove_path):
	input_file = open(glove_path)
	word_embedding = {}

	for line in input_file:
		word_vec = line.split()
		word_embedding[word_vec[0]] = np.asarray(word_vec[1:], dtype='float32')

	input_file.close()
	return word_embedding


# get dict: {word: index}, word index is start from 1
def get_word_index(num_words, skip_top=0, path=sys.path[0]+'/data/imdb_word_index.json'):
	input_file = open(path)
	original_word_index = json.load(input_file)
	input_file.close()

	word_index = {}
	for (word, i) in original_word_index.items():
		i = int(i)
		if (i >= skip_top and i < num_words):
			word_index[word] = i

	return word_index


# get dict: {index: embedding}
def get_index_embedding(word_index, word_embedding):
	index_embedding = np.zeros((len(word_index)+1, len(word_embedding.itervalues().next() )))

	for (word, i) in word_index.items():
		embedding = word_embedding.get(word)
		if embedding is not None:
			index_embedding[i] = embedding

	print('index_embedding.shape: ', index_embedding.shape)
	return index_embedding


def imdb_run(index_embedding, num_words=5000, max_len = 500):
	(x_train, y_train), (x_test, y_test) = load_data(num_words=num_words)
	x_train = sequence.pad_sequences(x_train, maxlen=max_len)
	x_test = sequence.pad_sequences(x_test, maxlen=max_len)

	embedding_len = 100
	model = Sequential()
	model.add(Embedding(num_words, embedding_len, input_length=max_len, weights=[index_embedding]))
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
					optimizer='adam',
					metrics=['accuracy'])

	print(model.summary())
	model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=2)
	score, acc = model.evaluate(x_test, y_test, verbose=0)

	print('Test score:', score)
	print('Test accuracy:', acc)



if __name__=='__main__':

	glove_path = sys.path[0]+'/data/glove.6B.100d.txt'
	word_embedding = get_word_embedding(glove_path)

	num_words = 5000
	word_index = get_word_index(num_words=num_words)
	index_embedding = get_index_embedding(word_index, word_embedding)

	imdb_run(index_embedding)

