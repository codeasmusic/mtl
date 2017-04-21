from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
from keras import regularizers, optimizers

import sys
import mytools.dataset as ds
import mytools.embedding as emb



def sst1_run(index_embedding, dataset, num_words=5000, embedding_len=100, max_len=50):

	(x_train, y_train), (x_test, y_test), (x_dev, y_dev) = ds.load_data(dataset, num_words, has_dev=True)
	x_train = sequence.pad_sequences(x_train, maxlen=max_len)
	x_test = sequence.pad_sequences(x_test, maxlen=max_len)
	x_dev = sequence.pad_sequences(x_dev, maxlen=max_len)

	num_classes = 5
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test, num_classes)
	y_dev = to_categorical(y_dev, num_classes)

	model = Sequential()
	model.add(Embedding(num_words, embedding_len, input_length=max_len, weights=[index_embedding]))
	model.add(LSTM(max_len, dropout=0.4, recurrent_dropout=0.2))
	model.add(Dense(num_classes, activation='sigmoid', 	# try softmax
					# kernel_regularizer=regularizers.l2(0.01),
					# activity_regularizer=regularizers.l1(0.01)
			))

	model.compile(loss='categorical_crossentropy',
					optimizer='adam',
					metrics=['accuracy'])

	print(model.summary())
	model.fit(x_train, y_train, epochs=5, batch_size=32,
			  validation_data=(x_dev, y_dev),  verbose=2)

	score, acc = model.evaluate(x_test, y_test, verbose=0)
	print('Test score:', score)
	print('Test accuracy:', acc)



if __name__=='__main__':

	home = sys.path[0] + '/data/'

	embedding_len = 100
	glove_file = home +'glove/glove.6B.' + str(embedding_len) +'d.txt'
	word_embedding = emb.get_word_embedding(glove_file)

	num_words = 7000
	word_index_file = home +'sst/sst1_word_index.json'
	word_index = emb.get_word_index(word_index_file, num_words)

	index_embedding = emb.get_index_embedding(word_index, word_embedding, num_words)
	dataset = home + 'sst/sst1.npz'

	max_len = 80
	sst1_run(index_embedding, dataset, num_words, embedding_len, max_len)