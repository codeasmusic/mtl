
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

from tools import preprocess
from tools import embedding


random.seed(113)


def rcnn_mtl(datasets, index_embedding, params):

	mtl_model, single_models, in_out_names = build_models(datasets, index_embedding, params)

	x_trains, y_trains, x_tests, y_tests = get_train_test(datasets, in_out_names)

	num_tasks = len(datasets)-1
	batch_size = params['batch_size']

	# iterations = get_iterations(x_trains, batch_size, params['epochs'])
	# sys.stdout.write('\n\niterations: {}'.format(iterations))

	iterations = 1100
	sys.stdout.write('\nspecific iterations: {}'.format(iterations))

	itera = 0
	batch_input = {}
	batch_output = {}
	
	while (itera < iterations):
		itera += 1
		if((itera % 100 == 0) or (itera > 900 and itera % 50 == 0)):
			sys.stdout.write('\ncurrent iteration: {}'.format(itera))
			evaluate(single_models, x_trains, y_trains, 'train')
			evaluate(single_models, x_tests, y_tests, 'test')

		generate_batch_data(batch_input, batch_output, batch_size, x_trains, y_trains)
		mtl_model.train_on_batch(batch_input, batch_output)

		# if(itera % iters_per_epoch == 0):
		# 	evaluate(single_models, x_trains, y_trains, 'train')
		# 	evaluate(single_models, x_tests, y_tests, 'test')	

	evaluate(single_models, x_trains, y_trains, 'train')
	return evaluate(single_models, x_tests, y_tests, 'test')


def build_models(datasets, index_embedding, params):
	loss = {}
	loss_weights = {}
	input_layers = []
	output_layers = []
	in_out_names = []
	single_models = {}

	shared_embedding = Embedding(input_dim=params['num_words'], 
								 output_dim=params['embedding_len'], 
							  	 # input_length=params['max_len'],
							  	 weights=[index_embedding],
							  	 name='shared_embedding')
	
	shared_reshape = Reshape((20, 10), name='shared_reshape')
	shared_conv1D = Conv1D(filters=params['filters'],
						 kernel_size=params['kernel_size'],
						 padding='valid', activation='relu',
						 strides=1, name='shared_conv1D')
	
	shared_maxPooling1D = MaxPooling1D(pool_size=params['pool_size'],
									   name='shared_maxPooling1D')

	shared_flatten = Flatten(name='shared_flatten')

	shared_dense = Dense(units=params['dense_units'], activation='relu',
						 name='shared_dense')

	dropout_conv = Dropout(0.25, name='dropout_conv')
	dropout_dense = Dropout(0.5, name='dropout_dense')


	for (index, ds) in enumerate(datasets):

		in_name = 'input_' + str(index)
		in_layer = Input(shape=(params['max_len_list'][index],), dtype='int32', name=in_name)
		input_layers.append(in_layer)

		mid_layer = shared_embedding(in_layer)
		# mid_layer = Bidirectional(LSTM(units=params['lstm_units'], return_sequences=True,
		# 								dropout=0.5, recurrent_dropout=0.5),
		# 						 name='bi_lstm'+ str(index))(mid_layer)

		mid_layer = Bidirectional(LSTM(units=params['lstm_units'], dropout=0.5, recurrent_dropout=0.5),
								 name='bi_lstm'+ str(index))(mid_layer)
		mid_layer = shared_reshape(mid_layer)
		# lstm_output = mid_layer

		mid_layer = shared_conv1D(mid_layer)
		mid_layer = shared_maxPooling1D(mid_layer)
		mid_layer = dropout_conv(mid_layer)

		# merge_output = concatenate([mid_layer, lstm_output], axis=1)

		mid_layer = shared_flatten(mid_layer)
		mid_layer = shared_dense(mid_layer)
		mid_layer = dropout_dense(mid_layer)

		# merge_output = shared_flatten(merge_output)
		# merge_output = shared_dense(merge_output)
		# merge_output = dropout_dense(merge_output)


		out_name = 'output_'+str(index)
		if ds['num_classes'] == 2:
			out_layer = Dense(units=1, activation='sigmoid', 
							  name=out_name)(mid_layer)

			# out_layer = Dense(units=1, activation='sigmoid', 
			# 				  name=out_name)(merge_output)

			loss[out_name] = 'binary_crossentropy'
		else:
			out_layer = Dense(units=ds['num_classes'], 
							  activation='softmax', name=out_name)(mid_layer)

			# out_layer = Dense(units=ds['num_classes'], 
			# 				  activation='softmax', name=out_name)(merge_output)

			loss[out_name] = 'categorical_crossentropy'
		output_layers.append(out_layer)

		# loss_weights[out_name] = 1.0 / len(datasets)
		loss_weights = params['loss_weights']

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

		# x_trains[in_name] = ds['x_train']
		# y_trains[out_name] = ds['y_train']
		# x_tests[in_name] = ds['x_test']
		# y_tests[out_name] = ds['y_test']

		rate = 1

		x_train = ds['x_train']
		y_train = ds['y_train']
		x_test = ds['x_test']
		y_test = ds['y_test']

		x_trains[in_name] = x_train[:len(x_train)/rate]
		y_trains[out_name] = y_train[:len(y_train)/rate]
		x_tests[in_name] = x_test[:len(x_test)/rate]
		y_tests[out_name] = y_test[:len(y_test)/rate]

		sys.stdout.write('\n{}, train: {}, test: {}'.format(in_name, 
						x_trains[in_name].shape, x_tests[in_name].shape))

	return x_trains, y_trains, x_tests, y_tests


def get_iterations(x_trains, batch_size, epochs):
	max_samples = 0
	for (in_name, samples) in x_trains.items():
		if(max_samples < len(samples)):
			max_samples = len(samples)

	iterations = (max_samples * 1.0 / batch_size) * epochs
	iterations = int(math.ceil(iterations))

	# iters_per_epoch = ((iterations+1)*1.0 / epochs)
	# iters_per_epoch = int(math.ceil(iters_per_epoch))

	return iterations


def generate_batch_data(batch_input, batch_output, batch_size, x_trains, y_trains):
	batch_input.clear()
	batch_output.clear()

	half_batch_size = batch_size / 2
	start = half_batch_size

	for (in_name, x_train), (out_name, y_train) in zip(x_trains.items(), y_trains.items()):
		assert (in_name[-1]==out_name[-1])
		end = len(x_train) - half_batch_size
		pivot = random.randint(start, end)

		batch_input[in_name] = x_train[pivot-half_batch_size: pivot+half_batch_size]
		batch_output[out_name] = y_train[pivot-half_batch_size: pivot+half_batch_size]


def evaluate(single_models, X, Y, flag):
	
	if(flag == 'train'):
		sys.stdout.write('\n========================================================')
	else:	
		sys.stdout.write('\n--------------------------------------------------------')
	sys.stdout.write('\n{}'.format(flag))

	average_acc = 0
	for index in xrange(len(single_models)):
		index = str(index)
		x = X['input_'+index]
		y = Y['output_'+index]
		model = single_models['model_'+index]

		loss, acc = model.evaluate(x, y, verbose=0)
		average_acc += acc
		# print(model.metrics_names)		# ['loss', 'acc']

		sys.stdout.write('\nmodel_{}:'.format(index))
		sys.stdout.write('\n\tloss: {}, accuracy: {}'.format(loss, acc))
		
	return (average_acc / 4.0)


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
			curr_acc = rcnn_mtl(datasets, index_embedding, params)
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



if __name__ == '__main__':

	home = sys.path[0] + '/data/'

	raw_files = [{'train': home+'sst1/sst1_train_label_sent.txt', 
				  'test': home+'sst1/sst1_test_label_sent.txt'},

				 {'train': home+'sst2/sst2_train_label_sent.txt', 
				  'test': home+'sst2/sst2_test_label_sent.txt'},

				 {'train': home+'subj/subj_train_label_sent.txt',
				  'test': home+'subj/subj_test_label_sent.txt'},

				  {'train': home+'imdb/imdb_train_label_sent.txt',
				  'test': home+'imdb/imdb_test_label_sent.txt'} ]

	num_classes_list = [5, 2, 2, 2]

	num_words=5000
	# max_len=64
	max_len_list = [50, 50, 60, 500]

	embedding_len = 100

	# datasets, word_index = preprocess.get_datasets(raw_files, num_classes_list,
	# 											   num_words, max_len_list)

	# glove_file = home +'glove/glove.6B.' + str(embedding_len) +'d.txt'
	# index_embedding = embedding.get_index_embedding(word_index, glove_file)

	# data = [datasets, word_index, index_embedding]
	# pickle.dump(data, open(home+'data.dat', 'w'))


	data = pickle.load(open(home+'data.dat'))
	datasets = data[0]
	word_index = data[1]
	index_embedding = data[2]

	num_words = index_embedding.shape[0]

	# loss_weights = {'output_0': 0.25, 'output_1': 0.25, 'output_2': 0.25, 'output_3': 0.25 }
	loss_weights = {'output_0': 0.55, 'output_1': 0.15, 'output_2': 0.15, 'output_3': 0.15 }

	params = {'num_words': num_words, 'max_len_list': max_len_list, 'embedding_len': embedding_len,
			  'batch_size': 64, 'epochs': 3, 'filters': 100, 'kernel_size': 5, 'pool_size': 4,
			  'lstm_units': 100, 'dense_units': 128, 'loss_weights': loss_weights}

	sys.stdout.write('\n--------------------------------------------------------------------')
	sys.stdout.write('\n{}'.format(params))
	sys.stdout.write('\n--------------------------------------------------------------------\n')

	start = datetime.datetime.now()
	average_acc = rcnn_mtl(datasets, index_embedding, params)
	end = datetime.datetime.now()
	sys.stdout.write('\nused time: {}\n'.format(end - start))


	# params['batch_size'] = 64
	# loss_weights_0 = {'output_0': 0.55, 'output_1': 0.15, 'output_2': 0.15, 'output_3': 0.15 }
	# loss_weights_1 = {'output_0': 0.15, 'output_1': 0.55, 'output_2': 0.15, 'output_3': 0.15 }
	# loss_weights_2 = {'output_0': 0.15, 'output_1': 0.15, 'output_2': 0.55, 'output_3': 0.15 }
	# loss_weights_3 = {'output_0': 0.15, 'output_1': 0.15, 'output_2': 0.15, 'output_3': 0.55 }

	# tuning_list = [ ('loss_weights', loss_weights_0, loss_weights_1, loss_weights_2, loss_weights_3)]
	# tuning_params(datasets, index_embedding, params, average_acc, tuning_list)

	
