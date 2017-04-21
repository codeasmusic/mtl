
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Bidirectional, LSTM
from keras.layers import Conv1D, MaxPooling1D

import sys
import pickle
import math
import random

from tools import preprocess
from tools import embedding



def rcnn_mtl(datasets, index_embedding, params):

	mtl_model, single_models, in_out_names = build_models(datasets, index_embedding, params)

	x_trains, y_trains, x_tests, y_tests = get_train_test(datasets, in_out_names)

	num_tasks = len(datasets)-1
	batch_size = params['batch_size']
	iterations = get_iterations(x_trains, batch_size, params['epochs'])
	# iterations = 1
	print('iterations: {}'.format(iterations))

	itera = 0
	batch_input = {}
	batch_output = {}
	
	# s_layer_dict = dict([(layer.name, layer) for layer in single_models['model_0'].layers])
	# print s_layer_dict['shared_dense'].get_weights()
	# print '-------------------------------------------------------------------------------'
	# m_layer_dict = dict([(layer.name, layer) for layer in mtl_model.layers])
	# print m_layer_dict['shared_dense'].get_weights()

	while (itera < iterations):
		itera += 1
		if(itera % 20 == 0):
			print('current iteration: {}'.format(itera))
		# print('current iteration: {}'.format(itera))

		generate_batch_data(batch_input, batch_output, batch_size, x_trains, y_trains)
		mtl_model.train_on_batch(batch_input, batch_output)

	evaluate(single_models, x_trains, y_trains)

	# print '###############################################################################'
	# print s_layer_dict['shared_dense'].get_weights()
	# print '-------------------------------------------------------------------------------'
	# print m_layer_dict['shared_dense'].get_weights()


def build_models(datasets, index_embedding, params):


	loss = {}
	loss_weights = {}
	input_layers = []
	output_layers = []
	in_out_names = []
	single_models = {}

	shared_embedding = Embedding(input_dim=params['num_words'], 
								 output_dim=params['embedding_len'], 
							  	 input_length=params['max_len'],
							  	 weights=[index_embedding],
							  	 name='shared_embedding')
	
	shared_conv1D = Conv1D(filters=params['filters'],
						 kernel_size=params['kernel_size'],
						 padding='valid', activation='relu',
						 strides=1, name='shared_conv1D')
	
	shared_maxPooling1D = MaxPooling1D(pool_size=params['pool_size'],
									   name='shared_maxPooling1D')

	shared_flatten = Flatten(name='shared_flatten')

	shared_dense = Dense(units=params['dense_units'], activation='relu',
						 name='shared_dense')

	
	print(params)
	print('----------------------------------------------------------------------------')


	for (index, ds) in enumerate(datasets):

		in_name = 'input_' + str(index)
		in_layer = Input(shape=(params['max_len'],), dtype='int32', name=in_name)
		input_layers.append(in_layer)

		mid_layer = shared_embedding(in_layer)
		mid_layer = Bidirectional(LSTM(units=params['lstm_units'], return_sequences=True),
								 name='bi_lstm'+ str(index))(mid_layer)

		# why use return_sequences=True? Because of the 3D dimension of CNN

		# print('----------------------------------------------------------------------------')
		# print('mid_layer: ', mid_layer.shape)
		# print('shared_conv1D: ', shared_conv1D.get_config())
		# print(dir(shared_conv1D))


		mid_layer = shared_conv1D(mid_layer)
		mid_layer = shared_maxPooling1D(mid_layer)
		mid_layer = shared_flatten(mid_layer)
		mid_layer = shared_dense(mid_layer)


		out_name = 'output_'+str(index)
		if ds['num_classes'] == 2:
			out_layer = Dense(units=1, activation='sigmoid', 
							  name=out_name)(mid_layer)
			loss[out_name] = 'binary_crossentropy'
		else:
			out_layer = Dense(units=ds['num_classes'], 
							  activation='softmax', name=out_name)(mid_layer)
			loss[out_name] = 'categorical_crossentropy'
		output_layers.append(out_layer)

		loss_weights[out_name] = 1.0 / len(datasets)
		in_out_names.append((in_name, out_name))

		curr_model = Model(inputs=in_layer, outputs=out_layer)
		curr_model.compile(loss=loss[out_name], optimizer='adam', metrics=['accuracy'])
		single_models['model_'+str(index)] = curr_model

	mtl_model = Model(inputs=input_layers, outputs=output_layers)
	mtl_model.compile(loss=loss, loss_weights=loss_weights, optimizer='adam')

	print(mtl_model.summary())

	return mtl_model, single_models, in_out_names


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

	return x_trains, y_trains, x_tests, y_tests


def get_iterations(x_trains, batch_size, epochs):
	max_samples = 0
	for (in_name, samples) in x_trains.items():
		if(max_samples < len(samples)):
			max_samples = len(samples)

	iterations = (max_samples * 1.0 / batch_size) * epochs
	return int(math.ceil(iterations))


def generate_batch_data(batch_input, batch_output, batch_size, x_trains, y_trains):
	batch_input.clear()
	batch_output.clear()

	half_batch_size = batch_size / 2
	start = half_batch_size

	for (in_name, x_train) in x_trains.items():
		end = len(x_train) - half_batch_size
		pivot = random.randint(start, end)
		batch_input[in_name] = x_train[pivot-half_batch_size: pivot+half_batch_size]

	for (out_name, y_train) in y_trains.items():
		end = len(y_train) - half_batch_size
		pivot = random.randint(start, end)
		batch_output[out_name] = y_train[pivot-half_batch_size: pivot+half_batch_size]


def evaluate(single_models, x_tests, y_tests):
	for index in xrange(len(single_models)):
		index = str(index)
		x_test = x_tests['input_'+index]
		y_test = y_tests['output_'+index]
		model = single_models['model_'+index]

		score, acc = model.evaluate(x_test, y_test, verbose=0)
		print('model_'+ index +':')
		print('Test score: {}, \tTest accuracy: {}'.format(score, acc))




if __name__ == '__main__':

	home = sys.path[0] + '/data/'

	raw_files = [{'train': home+'sst1/train_sent_label.txt', 
				  'test': home+'sst1/test_sent_label.txt'},

				 {'train': home+'sst2/bi_train_sent_label.txt', 
				  'test': home+'sst2/bi_test_sent_label.txt'}]

	num_classes_list = [5, 2]

	num_words=5000
	max_len=80
	embedding_len = 100

	# datasets, word_index = preprocess.get_datasets(raw_files, num_classes_list,
	# 											   num_words, max_len)

	# glove_file = home +'glove/glove.6B.' + str(embedding_len) +'d.txt'
	# index_embedding = embedding.get_index_embedding(word_index, glove_file)

	# data = [datasets, word_index, index_embedding]
	# pickle.dump(data, open(home+'data.dat', 'w'))


	data = pickle.load(open(home+'data.dat'))
	datasets = data[0]
	word_index = data[1]
	index_embedding = data[2]

	num_words = len(word_index)

	params = {'num_words': num_words, 'max_len': max_len, 'embedding_len': embedding_len,
			  'batch_size': 32, 'epochs': 3, 'filters': 64, 'kernel_size': 10, 'pool_size': 4,
			  'lstm_units': 90, 'dense_units': 128}

	rcnn_mtl(datasets, index_embedding, params)
