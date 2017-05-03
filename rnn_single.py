
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Bidirectional, LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import concatenate, Reshape

import sys
import pickle
import math
import random
import datetime
import numpy as np

from tools import preprocess
from tools import embedding



def rnn_single(processed_datasets, index_embedding, params):
	start = datetime.datetime.now()

	x_trains, y_trains, x_tests, y_tests = processed_datasets
	in_name = params['in_name']
	out_name = params['out_name']
	x_train = x_trains[in_name]
	y_train = y_trains[out_name]
	x_test = x_tests[in_name]
	y_test = y_tests[out_name]

	single_model = build_models(params, index_embedding)
	print(single_model.summary())

	iterations = 1100
	sys.stdout.write('\nspecific iterations: {}'.format(iterations))

	itera = 0
	batch_size = params['batch_size']

	while (itera < iterations):
		itera += 1
		if (itera % 100 == 0):
			sys.stdout.write('\n\ncurrent iteration: {}'.format(itera))
			evaluate(single_model, x_train, y_train, 'train')
			evaluate(single_model, x_test, y_test, 'test')

		batch_input, batch_output = generate_batch_data(batch_size, x_train, y_train)
		single_model.train_on_batch(batch_input, batch_output)

	evaluate(single_model, x_train, y_train, 'train')
	average_acc = evaluate(single_model, x_test, y_test, 'test')

	end = datetime.datetime.now()
	sys.stdout.write('\nused time: {}\n'.format(end - start))
	return average_acc


def process(datasets):
	in_out_names = []

	for index in xrange(len(datasets)):
		in_name = 'input_' + str(index)
		out_name = 'output_'+str(index)
		in_out_names.append((in_name, out_name))

	x_trains, y_trains, x_tests, y_tests = get_train_test(datasets, in_out_names)
	processed_datasets = (x_trains, y_trains, x_tests, y_tests)

	return [processed_datasets, in_out_names]


def build_models(params, index_embedding):

	in_layer = Input(shape=(params['max_len'],), dtype='int32')

	mid_layer = Embedding(input_dim=params['num_words'], 
						  output_dim=params['embedding_len'], 
						  weights=[index_embedding])(in_layer)

	# mid_layer = LSTM(params['lstm_output_dim'], return_sequences=True,
	# 						dropout=0.5, recurrent_dropout=0.5)(mid_layer)

	mid_layer = Bidirectional(LSTM(params['lstm_output_dim'], return_sequences=True,
							dropout=0.5, recurrent_dropout=0.5)) (mid_layer)

	mid_layer = Dense(params['dense_units'])(mid_layer)
	mid_layer = Dropout(0.3)(mid_layer)

	mid_layer = Flatten()(mid_layer)


	if (params['num_class'] == 2):
		loss = 'binary_crossentropy'
		out_layer = Dense(units=1, activation='sigmoid')(mid_layer)
	else:
		loss = 'categorical_crossentropy'
		out_layer = Dense(units=params['num_class'], activation='softmax')(mid_layer)

	single_model = Model(inputs=in_layer, outputs=out_layer)
	single_model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])

	return single_model


def get_train_test(datasets, in_out_names):
	x_trains = {}
	y_trains = {}
	x_tests = {}
	y_tests = {}

	for (index, ds) in enumerate(datasets):
		in_name = in_out_names[index][0]
		out_name = in_out_names[index][1]

		x_trains[in_name] = ds['x_train']
		y_trains[out_name] = ds['y_train']
		x_tests[in_name] = ds['x_test']
		y_tests[out_name] = ds['y_test']

		sys.stdout.write('\n{}, train: {}, test: {}'.format(in_name, 
						x_trains[in_name].shape, x_tests[in_name].shape))

	sys.stdout.write('\n\n')
	return x_trains, y_trains, x_tests, y_tests


def generate_batch_data(batch_size, x_train, y_train):

	half_batch_size = batch_size / 2
	start = half_batch_size
	end = len(x_train) - half_batch_size
	pivot = random.randint(start, end)

	batch_input = x_train[pivot-half_batch_size: pivot+half_batch_size]
	batch_output = y_train[pivot-half_batch_size: pivot+half_batch_size]
	return batch_input, batch_output


def evaluate(single_model, X, Y, flag):
	
	if(flag == 'train'):
		sys.stdout.write('\n========================================================')
	else:	
		sys.stdout.write('\n--------------------------------------------------------')
	sys.stdout.write('\n{}'.format(flag))

	loss, acc = single_model.evaluate(X, Y, verbose=0)
	sys.stdout.write('\n\tloss: {}, accuracy: {}'.format(loss, acc))


def tuning_params(datasets, index_embedding, params, average_acc, tuning_list):

	for param_tuple in tuning_list:
		param_name = param_tuple[0]
		param_values = param_tuple[1:]

		for new_value in param_values:
			old_value = params[param_name]
			if(new_value == old_value):
				continue

			params[param_name] = new_value
			sys.stdout.write('\n__________________________________________________')
			sys.stdout.write('\nparam: {}, value: {}'.format(param_name, new_value))
			sys.stdout.write('\n__________________________________________________')

			start = datetime.datetime.now()
			curr_acc = rnn_single(datasets, index_embedding, params)
			end = datetime.datetime.now()
			sys.stdout.write('\nused time: {}'.format(end - start))

			if((curr_acc-average_acc) > 0.005):
				average_acc = curr_acc
				sys.stdout.write('\n==================================================================')
				sys.stdout.write('\nimprove, average acc: {}'.format(average_acc))
				sys.stdout.write('\n------------------------------------------------------------------')
				sys.stdout.write('\n{}'.format(params))
				sys.stdout.write('\n==================================================================')
			else:
				params[param_name] = old_value

			sys.stdout.flush()


def print_params(params):
	sys.stdout.write('\n--------------------------------------------------------------------')
	sys.stdout.write('\n{}'.format(params))
	sys.stdout.write('\n--------------------------------------------------------------------\n')



if __name__ == '__main__':

	home = sys.path[0] + '/data/'

	raw_files = [
				 {'train': home+'sst1/sst1_train_label_sent.txt', 
				  'test': home+'sst1/sst1_test_label_sent.txt'},

				 # {'train': home+'sst2/sst2_train_label_sent.txt', 
				 #  'test': home+'sst2/sst2_test_label_sent.txt'},

				 # {'train': home+'subj/subj_train_label_sent.txt',
				 #  'test': home+'subj/subj_test_label_sent.txt'},

				 #  {'train': home+'imdb/imdb_train_label_sent.txt',
				 #  'test': home+'imdb/imdb_test_label_sent.txt'} 
				]

	task_name='sst1'
	num_words=8000
	embedding_len = 200
	# num_classes_list = [5, 2, 2, 2]
	# max_len_list = [50, 50, 60, 300]
	# min_freq_list = [2, 2, 2, 5]

	num_classes_list = [5]
	max_len_list = [60]
	min_freq_list = [1]
	data_file = home+'data_'+task_name+'.dat'


	# datasets, word_index = preprocess.get_datasets(raw_files, num_words, 
	# 											   num_classes_list, min_freq_list, max_len_list)
	# glove_file = home +'glove/glove.6B.' + str(embedding_len) +'d.txt'
	# index_embedding = embedding.get_index_embedding(word_index, glove_file)
	# data = (datasets, word_index, index_embedding)
	# pickle.dump(data, open(data_file, 'w'))


	data = pickle.load(open(data_file))
	datasets, word_index, index_embedding = data

	processed_datasets, in_out_names = process(datasets)

	params = {  
				'lstm_output_dim': 64,
				'dense_units': 128,
				'batch_size': 64, 
				'num_words': index_embedding.shape[0], 
				'max_len': max_len_list[0], 
				'embedding_len': embedding_len,
				'num_class': num_classes_list[0],
				'in_name': in_out_names[0][0],
				'out_name': in_out_names[0][1]
			}

	average_acc = rnn_single(processed_datasets, index_embedding, params)
	
