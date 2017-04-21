
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM

import sys
import mytools.dataset as ds
import mytools.embedding as emb



def subj_run(index_embedding, dataset, num_words=5000, embedding_len=100, max_len=50):

	(x_train, y_train), (x_test, y_test) = ds.load_data(dataset, num_words)
	x_train = sequence.pad_sequences(x_train, maxlen=max_len)
	x_test = sequence.pad_sequences(x_test, maxlen=max_len)

	model = Sequential()
	model.add(Embedding(num_words, embedding_len, input_length=max_len, weights=[index_embedding]))
	model.add(LSTM(max_len, dropout=0.5, recurrent_dropout=0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
					optimizer='adam',
					metrics=['accuracy'])

	print(model.summary())
	model.fit(x_train, y_train, epochs=4, batch_size=50, verbose=2)
	score, acc = model.evaluate(x_test, y_test, verbose=0)

	print('Test score:', score)
	print('Test accuracy:', acc)



if __name__=='__main__':

	home = sys.path[0]

	embedding_len = 200
	glove_file = home +'/data/glove/glove.6B.' + str(embedding_len) +'d.txt'
	word_embedding = emb.get_word_embedding(glove_file)

	num_words = 5000
	word_index_file = home +'/data/subj/subj_word_index.json'
	word_index = emb.get_word_index(word_index_file, num_words)

	index_embedding = emb.get_index_embedding(word_index, word_embedding, num_words)
	dataset = home + '/data/subj/subj.npz'

	max_len = 80
	subj_run(index_embedding, dataset, num_words, embedding_len, max_len)