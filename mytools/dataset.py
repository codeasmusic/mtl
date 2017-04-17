import numpy as np


def load_data(dataset, num_words, skip_top=0):
	print('Loading data...', num_words)

	input_file = np.load(dataset)
	x_train = input_file['x_train']	# 2D array, each row is [index of reivew's words]
	y_train = input_file['y_train']
	x_test = input_file['x_test']
	y_test = input_file['y_test']
	input_file.close()

	x_train = [filter(lambda w: w >= skip_top and w < num_words, x) for x in x_train]
	x_test = [filter(lambda w: w >= skip_top and w < num_words, x) for x in x_test]
	
	return (x_train, y_train), (x_test, y_test)


	