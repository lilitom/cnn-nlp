from __future__ import print_function
from __future__ import division

import string

import datetime
import numpy as np
np.random.seed(0123)  # for reproducibility

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder


def makeClean(text, numWords):
	line_split = text.split()
	tokens_text = line_split[0:numWords]
	tokens_text = [w.lower() for w in tokens_text]  # convert to lower case
	return tokens_text


def createModel(args):
	from keras.models import Model
	from keras.optimizers import SGD
	from keras.layers import Input, Dense, Dropout, Flatten
	from keras.layers.convolutional import Convolution1D, MaxPooling1D
	from keras.layers.normalization import BatchNormalization

	##### PARAMETERS #####

	deepFlag = args[0] # for 0) for CNN , 1) for Deep CNN

	maxlen = args[1] # 1300 #Max character length of text
	nb_epoch = args[2] # 20
	batch_size = args[3] # 64

	z1 = args[4] # 1024
	z2 = args[5] # 1024

	path = args[6]

	######################

	print('Loading data...')

	import csv
	X_train = []
	Y_train = []

	with open(path + '/data/amazon_review_polarity_csv/train.csv', 'r') as f:
		reader = csv.reader(f)
		for line in reader:
			X_train.append(line[1].strip() + line[2].strip())
			Y_train.append(int(line[0].strip()))

	X_train = np.array(X_train,dtype=object)
	Y_train = np.array(Y_train)

	X_test = []
	Y_test = []

	with open(path + '/data/amazon_review_polarity_csv/test.csv', 'r') as f:
	    reader = csv.reader(f)
	    for line in reader:
	        X_test.append(line[1].strip() + line[2].strip())
	        Y_test.append(int(line[0].strip()))

	X_test = np.array(X_test,dtype=object)
	Y_test = np.array(Y_test)


	X_train_clean = []

	for text in X_train:
	    X_train_clean.append(" ".join(makeClean(text,200)))

	X_train = np.array(X_train_clean,dtype=object)
	del X_train_clean

	X_test_clean = []

	for text in X_test:
	    X_test_clean.append(" ".join(makeClean(text,200)))

	X_test = np.array(X_test_clean,dtype=object)
	del X_test_clean

	enc = OneHotEncoder()
	Y_train = enc.fit_transform(Y_train[:, np.newaxis]).toarray()
	Y_test = enc.fit_transform(Y_test[:, np.newaxis]).toarray()

	##### PROCESS DATA ####

	### create vocab
	print('Get characters...')
	alphabet = (list(string.ascii_lowercase) + list(string.digits) + [' '] +
	            list(string.punctuation) + ['\n'])
	chars = set(alphabet)

	#chars = list(' '.join(X_train) + ' '.join(X_test))
	#chars = set(chars)

	vocab_size = len(chars)
	print('total chars:', len(chars))
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))

	### enocde data
	print('Vectorization...')
	X_train_char = np.zeros((len(X_train), maxlen, len(chars)), dtype=np.bool)
	for i, sentence in enumerate(X_train):
	    for t, char in enumerate(sentence):
	        X_train_char[i, t, char_indices[char]] = 1

	X_test_char = np.zeros((len(X_test), maxlen, len(chars)), dtype=np.bool)
	for i, sentence in enumerate(X_test):
	    for t, char in enumerate(sentence):
	        X_test_char[i, t, char_indices[char]] = 1

	print('train shape: ',X_train_char.shape)
	print('test shape: ',X_test_char.shape)

	##### SETUP CNN #####
	main_input = Input(shape=(maxlen,vocab_size), name='main_input')

	if deepFlag == 0: # Zhang et al. (2015), Character-level CNNs for Text Classification

	    conv = Convolution1D(nb_filter=256, filter_length=7,
	                         border_mode='valid', activation='relu',
	                         input_shape=(maxlen, vocab_size))(main_input)
	    conv = MaxPooling1D(pool_length=3)(conv)

	    conv1 = Convolution1D(nb_filter=256, filter_length=7,
	                          border_mode='valid', activation='relu')(conv)
	    conv1 = MaxPooling1D(pool_length=3)(conv1)

	    conv2 = Convolution1D(nb_filter=256, filter_length=3,
	                          border_mode='valid', activation='relu')(conv1)

	    conv3 = Convolution1D(nb_filter=256, filter_length=3,
	                          border_mode='valid', activation='relu')(conv2)

	    conv4 = Convolution1D(nb_filter=256, filter_length=3,
	                          border_mode='valid', activation='relu')(conv3)

	    conv5 = Convolution1D(nb_filter=256, filter_length=3,
	                          border_mode='valid', activation='relu')(conv4)
	    conv5 = MaxPooling1D(pool_length=3)(conv5)
	    conv5 = Flatten()(conv5)

	    #Two dense layers with dropout of .5
	    z = Dropout(0.5)(Dense(z1, activation='relu')(conv5))
	    z = Dropout(0.5)(Dense(z2, activation='relu')(z))

	else: # Alexis Conneau et al. (2016), Very Deep CNN For NLP.
	    # Depth is 9 layers + max pooling

	    #All the convolutional layers...
	    conv = Convolution1D(nb_filter=64, filter_length=3,
	                     border_mode='same', activation='relu',
	                     input_shape=(maxlen, vocab_size))(main_input)
	    conv = BatchNormalization()(conv)
	    conv = Dropout(0.1)(conv)

	    conv1 = Convolution1D(nb_filter=64, filter_length=3,
	                          border_mode='same', activation='relu')(conv)
	    conv1 = Convolution1D(nb_filter=64, filter_length=3,
	                          border_mode='same', activation='relu')(conv1)
	    conv1 = MaxPooling1D(pool_length=3,stride=2)(conv1)
	    conv1 = BatchNormalization()(conv1)
	    conv1 = Dropout(0.1)(conv1)

	    conv2 = Convolution1D(nb_filter=128, filter_length=3,
	                          border_mode='same', activation='relu')(conv1)
	    conv2 = Convolution1D(nb_filter=128, filter_length=3,
	                          border_mode='same', activation='relu')(conv2)
	    conv2 = MaxPooling1D(pool_length=3,stride=2)(conv2)
	    conv2 = BatchNormalization()(conv2)
	    conv2 = Dropout(0.1)(conv2)

	    conv3 = Convolution1D(nb_filter=256, filter_length=3,
	                          border_mode='same', activation='relu')(conv2)
	    conv3 = Convolution1D(nb_filter=256, filter_length=3,
	                          border_mode='same', activation='relu')(conv3)
	    conv3 = MaxPooling1D(pool_length=3,stride=2)(conv3)
	    conv3 = BatchNormalization()(conv3)
	    conv3 = Dropout(0.1)(conv3)

	    conv4 = Convolution1D(nb_filter=512, filter_length=3,
	                          border_mode='same', activation='relu')(conv3)
	    conv4 = Convolution1D(nb_filter=512, filter_length=3,
	                          border_mode='same', activation='relu')(conv4)
	    conv4 = MaxPooling1D(pool_length=3,stride=2)(conv4)
	    conv4 = BatchNormalization()(conv4)
	    conv4 = Dropout(0.1)(conv4)

	    conv4 = Flatten()(conv4)

	    #Two dense layers with dropout of .5
	    z = Dense(z1, activation='relu')(conv4) # z1
	    z = Dense(z2, activation='relu')(z) # z2


	#Output dense layer with softmax activation
	out = Dense(Y_train.shape[1], activation='softmax')(z)
	model = Model(input=main_input, output=out)
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)
	model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

	##### RUN MODEL ####
	history = model.fit(X_train_char,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_test_char, Y_test))
	score, acc = model.evaluate(X_test_char, Y_test, batch_size=batch_size)

	val_acc = history.history['val_acc']
	val_acc_max = np.amax(val_acc)
	val_acc_max_ep = np.argmax(val_acc)

	print('Best epoch:', val_acc_max_ep+1)
	print('Best test error:', 1-val_acc_max)
