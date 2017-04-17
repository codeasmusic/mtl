# encoding=utf-8
import sys
import json
import numpy as np
from keras.preprocessing import text


def preprocess_subj(raw_files, dataset_outfile, word_index_outfile):
	subj_infile = raw_files['subj']
	obj_infile = raw_files['obj']

	word_count = {}
	subj_sequences, word_count = get_sequences(subj_infile, word_count)
	obj_sequences, word_count = get_sequences(obj_infile, word_count)

	word_index = get_word_index(word_count, word_index_outfile)
	subj_index_seqs = get_index_sequences(subj_sequences, word_index)
	obj_index_seqs = get_index_sequences(obj_sequences, word_index)

	generate_train_test(subj_index_seqs, obj_index_seqs, dataset_outfile)



def get_sequences(raw_file, word_count):
	raw_sequences = []
	input_file = open(raw_file)
	
	for line in input_file:
		word_seq = text.text_to_word_sequence(line)
		raw_sequences.append(word_seq)

		for w in word_seq:
			if w in word_count:
				word_count[w] += 1
			else:
				word_count[w] = 1
	input_file.close()
	return raw_sequences, word_count
	

# index is start from 1
def get_word_index(word_count, word_index_outfile):
	sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

	word_index = {}
	for (word, count) in sorted_word_count:
		word_index[word] = len(word_index) + 1

	with open(word_index_outfile, 'w') as fp:
		json.dump(word_index, fp)

	return word_index


def get_index_sequences(raw_sequences, word_index):
	return [[word_index[w] for w in word_seq] for word_seq in raw_sequences]


def generate_train_test(subj_seqs, obj_seqs, outfile, test_num=1000, seed=113):
	assert (len(subj_seqs) > test_num and len(obj_seqs) > test_num), \
			"No enough data to be extracted as test data."

	x_train = np.concatenate([subj_seqs[test_num:], obj_seqs[test_num:]])
	y_train = np.concatenate([np.zeros(len(subj_seqs)-test_num),
								np.ones(len(obj_seqs)-test_num)])

	x_test = np.concatenate([subj_seqs[:test_num], obj_seqs[:test_num]])
	y_test = np.concatenate([np.zeros(test_num), np.ones(test_num)])

	np.random.seed(seed)
	np.random.shuffle(x_train)
	np.random.seed(seed)
	np.random.shuffle(y_train)

	np.random.seed(seed * 2)
	np.random.shuffle(x_test)
	np.random.seed(seed * 2)
	np.random.shuffle(y_test)

	np.savez(outfile, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__=='__main__':

	home = sys.path[0] + '/../data/subj/'

	raw_files = {'subj': home + 'subj_5000.txt', 'obj': home + 'obj_5000.txt'}

	dataset_outfile = home + 'subj.npz'
	word_index_outfile = home + 'subj_word_index.json'

	preprocess_subj(raw_files, dataset_outfile, word_index_outfile)
