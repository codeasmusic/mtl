
import sys
import numpy as np

from keras.preprocessing import text, sequence
from keras.utils.np_utils import to_categorical


def get_datasets(raw_files, num_classes_list, num_words, max_len):

	train_test_list = []
	global_word_count = {}

	for raw_file in raw_files:
		train_file = raw_file['train']
		test_file = raw_file['test']

		word_count = {}
		y_train, train_seq = get_text_sequences(train_file, word_count)
		y_test, test_seq = get_text_sequences(test_file, word_count)

		insert_to_global(word_count, num_words, global_word_count)
		train_test_list.append((train_seq, y_train, test_seq, y_test))

	word_index = get_word_index(global_word_count)

	datasets = []
	for (index, train_test) in enumerate(train_test_list):
		train_seq = train_test[0]
		test_seq = train_test[2]
		x_train = get_index_sequences(train_seq, word_index)
		x_test = get_index_sequences(test_seq, word_index)

		x_train = sequence.pad_sequences(x_train, maxlen=max_len)
		x_test = sequence.pad_sequences(x_test, maxlen=max_len)

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


def get_word_index(word_count):
	sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

	word_index = {}
	for (word, count) in sorted_word_count:
		word_index[word] = len(word_index)

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

	# print('global_words ', len(word_index))
	# outfile = open('word_index', 'w')
	# for (word, index) in word_index.items():
	# 	outfile.write(word + ':' + str(index) + '\n')
	# outfile.close()



if __name__ == '__main__':
	home = sys.path[0] + '/data/'

	raw_files = [{'train': home+'sst1/train_sent_label.txt', 
				  'test': home+'sst1/test_sent_label.txt'},

				 {'train': home+'sst2/bi_train_sent_label.txt', 
				  'test': home+'sst2/bi_test_sent_label.txt'}]

	num_classes_list = [5, 2]

	get_datasets(raw_files, num_classes_list, num_words=3000, max_len=100)
