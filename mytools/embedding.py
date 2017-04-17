import json
import numpy as np


# get dict: {word: embedding}
def get_word_embedding(glove_file):
	input_file = open(glove_file)
	word_embedding = {}

	for line in input_file:
		word_vec = line.split()
		word_embedding[word_vec[0]] = np.asarray(word_vec[1:], dtype='float32')

	input_file.close()
	return word_embedding


# get dict: {word: index}, word index is start from 1
def get_word_index(word_index_file, num_words, skip_top=0):
	input_file = open(word_index_file)
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


	