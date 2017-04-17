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
import mytools.dataset as ds
import mytools.embedding as emb



def imdb_run(index_embedding, dataset, num_words=5000, max_len = 500):

	(x_train, y_train), (x_test, y_test) = ds.load_data(dataset, num_words)
	x_train = sequence.pad_sequences(x_train, maxlen=max_len)
	x_test = sequence.pad_sequences(x_test, maxlen=max_len)

	embedding_len = 200
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

	home = sys.path[0]

	glove_file = home +'/data/glove/glove.6B.200d.txt'
	word_embedding = emb.get_word_embedding(glove_file)

	num_words = 5000
	word_index_file = home +'/data/imdb/imdb_word_index.json'
	word_index = emb.get_word_index(word_index_file, num_words)

	index_embedding = emb.get_index_embedding(word_index, word_embedding)
	dataset = home + '/data/imdb/imdb.npz'

	imdb_run(index_embedding, dataset, num_words)