
import numpy as np


# get dict: {index: embedding}
def get_index_embedding(word_index, glove_file):
	word_embedding = get_word_embedding(glove_file)

	index_embedding = np.zeros(( len(word_index), 
								 len(word_embedding.itervalues().next()) ))

	for (word, i) in word_index.items():
		embedding = word_embedding.get(word)
		if embedding is not None:
			index_embedding[i] = embedding

	print('index_embedding.shape: ', index_embedding.shape)
	return index_embedding


# get dict: {word: embedding}
def get_word_embedding(glove_file):
	input_file = open(glove_file)
	word_embedding = {}

	for line in input_file:
		word_vec = line.split()
		word_embedding[word_vec[0]] = np.asarray(word_vec[1:], dtype='float32')

	input_file.close()
	return word_embedding