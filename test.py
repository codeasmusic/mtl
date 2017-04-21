
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Dropout

import sys
import pickle

from tools import preprocess
from tools import embedding



def rcnn_mtl(datasets, num_words, max_len, embedding_len, index_embedding):
	assert len(datasets) > 0, 'len(datasets): ' + len(datasets)
	assert len(index_embedding) > 0, 'len(index_embedding): ' + len(index_embedding)
	assert (num_words > 0 and max_len > 0 and embedding_len > 0), 'negative parameters'

	input_layers = []
	output_layers = []

	x_trains = {}
	y_trains = {}
	x_tests = {}
	y_tests = {}

	loss = {}
	loss_weights = {}
	shared_layer = LSTM(units=128, name='shared')

	mtl_models = []

	for (index, ds) in enumerate(datasets):

		in_name = 'input_' + str(index)
		in_layer = Input(shape=(max_len,), dtype='int32', name=in_name)
		input_layers.append(in_layer)

		mid_layer = Embedding(input_dim=num_words, output_dim=embedding_len, name='emb_' + in_name, 
							  input_length=max_len, weights=[index_embedding])(in_layer)
		mid_layer = shared_layer(mid_layer)

		out_name = 'output_'+str(index)
		
		if ds['num_classes'] > 2:
			out_layer = Dense(units=ds['num_classes'], activation='softmax', name=out_name)(mid_layer)
			loss[out_name] = 'categorical_crossentropy'
		else:
			out_layer = Dense(units=1, activation='sigmoid', name=out_name)(mid_layer)
			loss[out_name] = 'binary_crossentropy'

		output_layers.append(out_layer)	
		loss_weights[out_name] = 1.0 / len(datasets)

		x_trains[in_name] = ds['x_train']
		y_trains[out_name] = ds['y_train']
		x_tests[in_name] = ds['x_test']
		y_tests[out_name] = ds['y_test']

		mtl_models.append(Model(inputs=in_layer, outputs=out_layer))
		mtl_models[index].compile(optimizer='adam', loss=loss[out_name], metrics=['accuracy'])

	# ---------------------------------------------------------------------------------------------

	mtl_model = Model(inputs=input_layers, outputs=output_layers)
	mtl_model.compile(optimizer='adam', loss=loss, loss_weights=loss_weights)

	layer_dict = dict([(layer.name, layer) for layer in mtl_model.layers])
	# print layer_dict['shared'].get_weights()
	# print mtl_model.layers[4].get_weights()

	print('-------------------------------------------------------------------')

	layer_dict_0 = dict([(layer.name, layer) for layer in mtl_models[0].layers])
	print layer_dict_0['shared'].get_weights()

	# score, acc = mtl_models[0].evaluate(x_tests['input_0'], y_tests['output_0'], verbose=0)
	# print('Test score:', score)
	# print('Test accuracy:', acc)

	# score, acc = mtl_models[1].evaluate(x_tests['input_1'], y_tests['output_1'], verbose=0)
	# print('Test score:', score)
	# print('Test accuracy:', acc)

	batch_input = {}
	for (in_name, x_train) in x_trains.items():
		batch_input[in_name] = x_train[:32]
		print(in_name, batch_input[in_name].shape)

	batch_output = {}
	for (out_name, y_train) in y_trains.items():
		batch_output[out_name] = y_train[:32]
		print(out_name, batch_output[out_name].shape)

	mtl_model.train_on_batch(batch_input, batch_output)	# ###
	# mtl_model.fit(batch_input, batch_output, epochs = 1, batch_size = 32, verbose=2)

	# mtl_models[0].train_on_batch(batch_input['input_0'], batch_output['output_0'])

	print('after train........................................................')

	# print layer_dict['shared'].get_weights()
	# print mtl_model.layers[4].get_weights()
	print('-------------------------------------------------------------------')
	print layer_dict_0['shared'].get_weights()

	

	# score, acc = mtl_models[0].evaluate(x_tests['input_0'], y_tests['output_0'], verbose=0)
	# print('Test score:', score)
	# print('Test accuracy:', acc)

	# score, acc = mtl_models[1].evaluate(x_tests['input_1'], y_tests['output_1'], verbose=0)
	# print('Test score:', score)
	# print('Test accuracy:', acc)


	# ---------------------------------------------------------------------------------------------
	# mtl_model.fit(x_trains, y_trains, epochs = 1, batch_size = 64, verbose=2)

	# mtl_model.evaluate(x_tests, y_tests, verbose=0)


if __name__ == '__main__':

	home = sys.path[0] + '/data/'

	raw_files = [{'train': home+'sst1/train_sent_label.txt', 
				  'test': home+'sst1/test_sent_label.txt'},

				 {'train': home+'sst2/bi_train_sent_label.txt', 
				  'test': home+'sst2/bi_test_sent_label.txt'}]

	num_classes_list = [5, 2]

	num_words=3000
	max_len=100
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

	rcnn_mtl(datasets, num_words, max_len, embedding_len, index_embedding)
