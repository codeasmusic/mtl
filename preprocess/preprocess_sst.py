
import sys
import json
import numpy as np
from keras.preprocessing import text


# ------------------generate label & sentence file-----------------

def sentencesSplit(datasetSents_file, datasetSplit_file, outfiles):
	firstLine = True
	sentId_splitId = {}
	
	split_file = open(datasetSplit_file)
	for line in split_file:
		if firstLine:
			firstLine = False
			continue
		line_parts = line.strip().split(',')
		sentId_splitId[line_parts[0]] = line_parts[1]
	split_file.close()

	firstLine = True
	sents_file = open(datasetSents_file)	
	fp = {	'1': open(outfiles['train'], 'w'),
			'2': open(outfiles['test'], 'w'),
			'3': open(outfiles['dev'], 'w')}

	for line in sents_file:
		if firstLine:
			firstLine = False
			continue
		line_parts = line.split('\t')
		sentId = line_parts[0]
		sentence = line_parts[1]

		splitId = sentId_splitId[sentId]
		fp[splitId].write(sentence)

	sents_file.close()
	fp['1'].close()
	fp['2'].close()
	fp['3'].close()


def sentencesLabel(sent_file, label_file, outfile):
	label_list = []
	lf = open(label_file)

	for line in lf:
		line_parts = line.split('(')
		label_list.append(line_parts[1].strip())
	lf.close()

	sf = open(sent_file)
	uf = open(outfile, 'w')

	line_num = 0
	for line in sf:
		uf.write(label_list[line_num] + '\t' + line)
		line_num += 1

	sf.close()
	uf.close()


def binarization(sent_label_file, outfile):
	slf = open(sent_label_file)
	uf = open(outfile, 'w')

	for line in slf:
		line_parts = line.split('\t')
		label = line_parts[0]
		sentence = line_parts[1]

		if(label == '2'):
			continue
		elif(int(label) < 2):
			uf.write('0' + '\t' + sentence)
		else:
			uf.write('1' + '\t' + sentence)

	slf.close()
	uf.close()



# ------------------generate npz file and word index file------------------

def preprocess_sst(raw_files, dataset_outfile, word_index_outfile):
	train_file = raw_files['train']
	test_file = raw_files['test']
	dev_file = raw_files['dev']

	word_count = {}
	y_train, train_seqs, word_count = get_sequences(train_file, word_count)
	y_test, test_seqs, word_count = get_sequences(test_file, word_count)
	y_dev, dev_seqs, word_count = get_sequences(dev_file, word_count)

	word_index = get_word_index(word_count, word_index_outfile)
	x_train = get_index_sequences(train_seqs, word_index)
	x_test = get_index_sequences(test_seqs, word_index)
	x_dev = get_index_sequences(dev_seqs, word_index)

	np.savez(dataset_outfile, x_train=np.asarray(x_train), y_train=np.asarray(y_train),
							  x_test=np.asarray(x_test), y_test=np.asarray(y_test),
							  x_dev=np.asarray(x_dev), y_dev=np.asarray(y_dev))


def get_sequences(raw_file, word_count):
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
	return label_list, raw_sequences, word_count


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




if __name__=='__main__':
	home = sys.path[0] + '/../data/sst/intermedia/'

	# datasetSents_file = home + 'datasetSentences.txt'
	# datasetSplit_file = home + 'datasetSplit.txt'
	# sent_outfiles = { 'train': home + 'train_sents.txt',
	# 				  'test' : home + 'test_sents.txt',
	# 				  'dev'  : home + 'dev_sents.txt'	}

	# sentencesSplit(datasetSents_file, datasetSplit_file, sent_outfiles)

	# label_files = { 'train': home + 'train.txt',
	# 			 	'test' : home + 'test.txt',
	# 			 	'dev'  : home + 'dev.txt'	}
	sent_label_outfiles = { 'train': home + 'train_sent_label.txt',
					 	 	'test' : home + 'test_sent_label.txt',
					 	 	'dev'  : home + 'dev_sent_label.txt'	}

	# sentencesLabel(sent_outfiles['train'], label_files['train'], sent_label_outfiles['train'])
	# sentencesLabel(sent_outfiles['test'], label_files['test'], sent_label_outfiles['test'])
	# sentencesLabel(sent_outfiles['dev'], label_files['dev'], sent_label_outfiles['dev'])

	bi_sent_label_outfiles = { 'train': home + 'bi_train_sent_label.txt',
						 	   'test' : home + 'bi_test_sent_label.txt',
						 	   'dev'  : home + 'bi_dev_sent_label.txt'	}

	# binarization(sent_label_outfiles['train'], bi_sent_label_outfiles['train'])
	# binarization(sent_label_outfiles['test'], bi_sent_label_outfiles['test'])
	# binarization(sent_label_outfiles['dev'], bi_sent_label_outfiles['dev'])


	# raw_files = bi_sent_label_outfiles
	# dataset_outfile = home + 'sst2.npz'
	# word_index_outfile = home + 'sst2_word_index.json'


	raw_files = sent_label_outfiles
	dataset_outfile = home + 'sst.npz'
	word_index_outfile = home + 'sst_word_index.json'

	preprocess_sst(raw_files, dataset_outfile, word_index_outfile)
