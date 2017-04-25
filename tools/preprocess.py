
import sys
import numpy as np

from keras.preprocessing import text, sequence
from keras.utils.np_utils import to_categorical


def get_datasets(raw_files, num_classes_list, num_words, max_len_list):

	train_test_list = []
	global_word_count = {}

	for raw_file in raw_files:
		train_file = raw_file['train']
		test_file = raw_file['test']

		word_count = {}
		y_train, train_seq = get_text_sequences(train_file, word_count)
		y_test, test_seq = get_text_sequences(test_file, word_count)

		seq_length_stat(test_seq, 5)

		insert_to_global(word_count, num_words, global_word_count)
		train_test_list.append((train_seq, y_train, test_seq, y_test))

	word_index = get_word_index(global_word_count)

	datasets = []
	for (index, train_test) in enumerate(train_test_list):
		train_seq = train_test[0]
		test_seq = train_test[2]
		x_train = get_index_sequences(train_seq, word_index)
		x_test = get_index_sequences(test_seq, word_index)

		x_train = sequence.pad_sequences(x_train, maxlen=max_len_list[index])
		x_test = sequence.pad_sequences(x_test, maxlen=max_len_list[index])

		y_train = np.asarray(train_test[1])
		y_test = np.asarray(train_test[3])
		num_classes = num_classes_list[index]

		if (num_classes > 2):
			y_train = to_categorical(y_train, num_classes)
			y_test = to_categorical(y_test, num_classes)

		datasets.append({'x_train': x_train, 'y_train': y_train,
						 'x_test': x_test, 'y_test': y_test,
						 'num_classes': num_classes})

	print_message(datasets, word_index)
	return datasets, word_index


def seq_length_stat(seqs, slices):
	max_length = 0
	for seq in seqs:
		if(max_length < len(seq)):
			max_length = len(seq)

	import math
	unit = math.ceil(1.0 * max_length / slices)

	unit_map = {}
	for i in xrange(slices):
		unit_map[(i+1)*unit] = 0

	for seq in seqs:
		for i in xrange(slices):
			curr_unit = (i+1)*unit
			if(len(seq) < curr_unit):
				unit_map[curr_unit] += 1
				break

	sorted_unit_map = sorted(unit_map.items(), key=lambda x: x[0])
	print('max length: {}'.format(max_length))
	print(sorted_unit_map)



def get_text_sequences(raw_file, word_count):
	label_list = []
	raw_sequences = []
	input_file = open(raw_file)
	
	for line in input_file:
		line_parts = line.strip().split('\t')
		label = line_parts[0]
		label_list.append(label)

		sentence = line_parts[1]
		word_seq = text.text_to_word_sequence(sentence)
		raw_sequences.append(word_seq)

		for w in word_seq:
			if w in word_count:
				word_count[w] += 1
			else:
				word_count[w] = 1
	input_file.close()
	return label_list, raw_sequences


def insert_to_global(word_count, num_words, global_word_count):
	sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

	for (word, count) in sorted_word_count[:num_words]:
		if word in global_word_count:
			global_word_count[word] += count
		else:
			global_word_count[word] = count


# index should start from 1, because pad_sequence will pad short sequence with 0.
def get_word_index(word_count):
	sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

	word_index = {}
	for (word, count) in sorted_word_count:
		word_index[word] = len(word_index) + 1

	return word_index


def get_index_sequences(raw_sequences, word_index):
	index_sequences = []
	for word_seq in raw_sequences:
		index_seq = []

		for w in word_seq:
			if w in word_index:
				index_seq.append(word_index[w])

		index_sequences.append(index_seq)

	return index_sequences


def print_message(datasets, word_index):
	msg = str(len(datasets)) + ' datasets, num_classes: ('
	for ds in datasets:
		msg += str(ds['num_classes']) + ', '
	msg = msg.rstrip(', ') + '), num_words: ' + str(len(word_index))
	print(msg)




if __name__ == '__main__':
	home = sys.path[0] + '/../data/'

	raw_files = [{'train': home+'sst1/train_label_sent.txt', 
				  'test': home+'sst1/test_label_sent.txt'},

				 {'train': home+'sst2/bi_train_label_sent.txt', 
				  'test': home+'sst2/bi_test_label_sent.txt'}]

	num_classes_list = [5, 2]

	datasets, word_index = get_datasets(raw_files, num_classes_list, num_words=1000, max_len=50)


	outfile = open(sys.path[0]+'/word_index.dat', 'w')
	for (word, index) in word_index.items():
		outfile.write(word + ':' + str(index) + '\n')
	outfile.close()
