
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.layers import concatenate, Reshape, add, maximum
from keras import optimizers, regularizers

import os
import sys
import pickle
import random
import datetime
import numpy as np

from keras.utils import plot_model

from tools import softmax_preprocess as preprocess
from tools import embedding
# from tools import RCNN_Attention


def rcnn_mtl(processed_datasets, index_embedding, params):
	start = datetime.datetime.now()

	x_trains, y_trains, x_tests, y_tests = processed_datasets
	mtl_model, single_models = build_models(params, index_embedding)
	print(mtl_model.summary())
	# plot_model(mtl_model, to_file='mtl_model.png', show_shapes=True)

	itera = 0
	batch_input = {}
	batch_output = {}
	batch_size = params['batch_size']
	iterations = params['iterations']
	sys.stdout.write('\ntotal iterations: {}'.format(iterations))

	while (itera < iterations):
		generate_batch_data(batch_input, batch_output, batch_size, x_trains, y_trains)
		mtl_model.train_on_batch(batch_input, batch_output)

		itera += 1
		if (itera > 200 and itera % 100 == 0):
			sys.stdout.write('\n\ncurrent iteration: {}'.format(itera))
			# evaluate(single_models, x_trains, y_trains, 'train')
			evaluate(single_models, x_tests, y_tests, 'test')
			sys.stdout.flush()

			if (itera >= 500):
				save_predictions(single_models, x_tests, params['prediction_path'])
				# save_models(single_models, params['save_model_path'])

	end = datetime.datetime.now()
	sys.stdout.write('\nused time: {}\n'.format(end - start))


def save_models(single_models, file_path):
	for index in xrange(len(single_models)):
		name = 'model_' + str(index)
		model = single_models[name]

		model_json = model.to_json()
		with open(file_path+name+'.json', 'w') as json_file:
			json_file.write(model_json)

		model.save_weights(file_path+name+'.h5')
		print('\nsave ' + name)


def build_models(params, index_embedding):
	loss = {}
	input_layers = []
	output_layers = []
	single_models = {}
	in_out_names = params['in_out_names']
	num_classes_list = params['num_classes_list']

	shared_embedding = Embedding(input_dim=params['num_words'], output_dim=params['embedding_len'],
							  	 weights=[index_embedding], name='global_embedding')

	shared_forward_lstm = LSTM(units=params['lstm_output_dim'], return_sequences=True,
							   dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'], 
							   name='shared_forward_lstm')

	shared_backward_lstm = LSTM(units=params['lstm_output_dim'], return_sequences=True,
							   dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'], 
							   name='shared_backward_lstm', go_backwards=True)

	shared_conv = Conv1D(filters=params['filters'], kernel_size=params['kernel_size'],
						 padding='same', activation='relu', strides=1, name='shared_conv')

	shared_dropout = Dropout(params['conv_dropout'], name='shared_dropout')

	shared_fw_tb = TimeDistributed(Dense(params['lstm_output_dim'], use_bias=False), name='shared_fw_tb')
	shared_bw_tb = TimeDistributed(Dense(params['lstm_output_dim'], use_bias=False), name='shared_bw_tb')
	shared_conv_tb = TimeDistributed(Dense(params['lstm_output_dim'], use_bias=False), name='shared_conv_tb')

	for index in xrange(len(in_out_names)):

		in_name = in_out_names[index][0]
		in_layer = Input(shape=(params['max_len_list'][index],), dtype='int32', name=in_name)
		input_layers.append(in_layer)

		specific_embedding = Embedding(input_dim=params['num_words'], output_dim=params['embedding_len'],
										weights=[index_embedding], name='local_embedding_'+str(index))

		emb_local = specific_embedding(in_layer)
		emb_global = shared_embedding(in_layer)

		emb_layer = concatenate([emb_local, emb_global], axis=2, name='emb_concat_'+str(index))

		fw_lstm_out = shared_forward_lstm(emb_layer)
		bw_lstm_out = shared_backward_lstm(emb_layer)
		conv_out = shared_conv(emb_layer)
		# conv_out = shared_dropout(conv_out)

		fw_tb_out = shared_fw_tb(fw_lstm_out)
		bw_tb_out = shared_bw_tb(bw_lstm_out)
		conv_tb_out = shared_conv_tb(conv_out)

		merge_out = add([fw_tb_out, conv_tb_out, bw_tb_out])

		# fw_lstm_out = Reshape((params['max_len_list'][index], params['lstm_output_dim'], 1),
		# 					  name = 'reshape_' + str(index) + '_0')(fw_lstm_out)
		# bw_lstm_out = Reshape((params['max_len_list'][index], params['lstm_output_dim'], 1),
		# 					  name = 'reshape_' + str(index) + '_1')(bw_lstm_out)
		# conv_out = Reshape((params['max_len_list'][index], params['lstm_output_dim'], 1),
		# 				      name='reshape_' + str(index) + '_2')(conv_out)

		# merge_out = concatenate([fw_lstm_out, conv_out, bw_lstm_out], axis=3, name='merge_'+str(index))
		# merge_out = TimeDistributed(Dense(1), name='time_distrib_' + str(index))(merge_out)
        #
		# merge_out = Reshape((params['max_len_list'][index], params['lstm_output_dim']),
		# 					name='reshape_' + str(index) + '_final')(merge_out)

		# bi_lstm = Bidirectional(LSTM(units=params['bi_lstm_output_dim'], return_sequences=True,
		# 					   		 dropout=params['bi_lstm_dropout'], recurrent_dropout=params['bi_lstm_dropout']),
		# 						name='bi_lstm' + str(index))
		# merge_out = bi_lstm(merge_out)

		# merge_out = Flatten(name='flatten_' + str(index))(merge_out)
		# merge_out = Dense(128, activation='sigmoid')(merge_out)
		# merge_out = Dropout(0.5)(merge_out)

		merge_out = LSTM(units=128, dropout=params['lstm_dropout'],
						 recurrent_dropout=params['lstm_dropout'], name='lstm_'+str(index))(merge_out)
		merge_out = Dense(128)(merge_out)
		merge_out = Dropout(0.5)(merge_out)


		out_name = in_out_names[index][1]
		num_class = num_classes_list[index]
		# if num_class == 2:
		# 	loss[out_name] = 'binary_crossentropy'
		# 	out_layer = Dense(units=1, activation='sigmoid',
		# 					  activity_regularizer=params['regularizers'][index], name=out_name)(merge_out)
		# else:
		loss[out_name] = 'categorical_crossentropy'
		out_layer = Dense(units=num_class, activation='softmax',
						  activity_regularizer=params['regularizers'][index], name=out_name)(merge_out)
		output_layers.append(out_layer)

		curr_model = Model(inputs=in_layer, outputs=out_layer)
		curr_model.compile(loss=loss[out_name], optimizer='rmsprop', metrics=['accuracy'])
		single_models['model_'+str(index)] = curr_model

	mtl_model = Model(inputs=input_layers, outputs=output_layers)
	mtl_model.compile(loss=loss, loss_weights=params['loss_weights'], optimizer='rmsprop')

	return mtl_model, single_models


def process(datasets):
	in_out_names = []

	for index in xrange(len(datasets)):
		in_name = 'input_' + str(index)
		out_name = 'output_'+str(index)
		in_out_names.append((in_name, out_name))

	x_trains, y_trains, x_tests, y_tests = get_train_test(datasets, in_out_names)
	processed_datasets = (x_trains, y_trains, x_tests, y_tests)

	return [processed_datasets, in_out_names]


def get_train_test(datasets, in_out_names, seed=109):
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

		np.random.seed(seed)
		np.random.shuffle(x_trains[in_name])
		np.random.seed(seed)
		np.random.shuffle(y_trains[out_name])
		np.random.seed(seed * 2)
		np.random.shuffle(x_tests[in_name])
		np.random.seed(seed * 2)
		np.random.shuffle(y_tests[out_name])

		sys.stdout.write('\n{}, train: {}, test: {}'.format(in_name,
							x_trains[in_name].shape, x_tests[in_name].shape))

	sys.stdout.write('\n\n')
	return x_trains, y_trains, x_tests, y_tests


def generate_batch_data(batch_input, batch_output, batch_size, x_trains, y_trains):
	batch_input.clear()
	batch_output.clear()

	for (in_name, x_train), (out_name, y_train) in zip(x_trains.items(), y_trains.items()):
		assert (in_name[-1] == out_name[-1])
		end = len(x_train) - batch_size
		pivot = random.randint(0, end)

		batch_input[in_name] = x_train[pivot: pivot+batch_size]
		batch_output[out_name] = y_train[pivot: pivot+batch_size]


def evaluate(single_models, X, Y, flag):

	if flag == 'train':
		sys.stdout.write('\n========================================================')
	else:
		sys.stdout.write('\n--------------------------------------------------------')
	sys.stdout.write('\n{}'.format(flag))

	for index in xrange(len(single_models)):
		index = str(index)
		x = X['input_'+index]
		y = Y['output_'+index]
		model = single_models['model_'+index]

		loss, acc = model.evaluate(x, y, verbose=0)
		sys.stdout.write('\nmodel_{}:'.format(index))
		sys.stdout.write('\n\tloss: {}, accuracy: {}'.format(loss, acc))
	sys.stdout.write('\n________________________________________________________')


def save_predictions(single_models, x_tests, file_path):
	predictions = []

	for index in xrange(len(single_models)):
		index = str(index)
		model = single_models['model_' + index]
		x = x_tests['input_' + index]

		predict_y = model.predict(x)
		predictions.append(predict_y)

	index = pickle.load(open('data/index'))
	pickle.dump(predictions, open(file_path + 'predictions_'+str(index)+'.dat', 'w'))
	pickle.dump(index+1, open('data/index', 'w'))


def check_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)



if __name__ == '__main__':

	home = sys.path[0] + '/data/'

	raw_files = [

		{'train': home + 'sst1/sst1_train_label_sent.txt',
		 'test': home + 'sst1/sst1_test_label_sent.txt'},

		{'train': home + 'sst2/sst2_train_label_sent.txt',
		 'test': home + 'sst2/sst2_test_label_sent.txt'},
		
		{'train': home+'subj/subj_train_label_sent.txt',
		 'test': home+'subj/subj_test_label_sent.txt'},

		{'train': home+'imdb/imdb_train_label_sent.txt',
		 'test': home+'imdb/imdb_test_label_sent.txt'},

	]

	embedding_len = 200
	num_classes_list = [5, 2, 2, 2]
	max_len_list = [100, 100, 100, 100]
	min_freq_list = [1, 1, 1, 20]

#	num_classes_list = [5, 2]
#	max_len_list = [75, 75]
#	min_freq_list = [1, 1]

	outfile_name = home + 'datasets.dat'

	datasets, word_index = preprocess.get_datasets(raw_files, num_classes_list,
													min_freq_list, max_len_list)
	glove_file = home +'glove/glove.6B.' + str(embedding_len) +'d.txt'
	index_embedding = embedding.get_index_embedding(word_index, glove_file)

	datasets, in_out_names = process(datasets)
	# pickle.dump(datasets, open(outfile_name, 'w'))

#	loss_weights = {'output_0': 0.5, 'output_1': 0.5}

	loss_weights = {'output_0': 0.25, 'output_1': 0.25,
					'output_2': 0.25, 'output_3': 0.25}

	regularizers_list = [regularizers.l2(0.0001), regularizers.l2(0.0001),
						 regularizers.l2(0.0001), regularizers.l2(0.0001)]

	params = {  
				'kernel_size': 3,
				'filters': 100,
				'lstm_output_dim': 100,
				# 'bi_lstm_output_dim': 32,

				'lstm_dropout': 0.5,
				'conv_dropout': 0.5,
				'bi_lstm_dropout': 0.5,

				'batch_size': 50,
				'iterations': 2000,
				'max_len_list': max_len_list,
				'embedding_len': embedding_len,
				'loss_weights': loss_weights, 
				'in_out_names': in_out_names,
				'regularizers': regularizers_list,
				'num_classes_list': num_classes_list,
				'num_words': index_embedding.shape[0],
				'save_model_path': sys.path[0] + '/model_folder/',
				'prediction_path': sys.path[0] + '/preds_folder/',
			}

	check_folder(params['save_model_path'])
	check_folder(params['prediction_path'])

	rcnn_mtl(datasets, index_embedding, params)
