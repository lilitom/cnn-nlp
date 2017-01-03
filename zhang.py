from __future__ import print_function
from __future__ import division

import string
import datetime

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization

np.random.seed(0123)


##### PARAMETERS #####

maxlen = options.maxlen
nb_epoch = options.num_epoch
batch_size = options.batch_size
z1 = options.z1
z2 = options.z2
train = options.train
test = options.test

######################

def makeClean(text, numWords):
	line_split = text.split()
	tokens_text = line_split[0:numWords]
	tokens_text = [w.lower() for w in tokens_text]
	return tokens_text

##################################
# LOAD DATA
#################################

print('Loading data...')

import csv
X_train = []
Y_train = []

with open('./data/'+train, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        X_train.append(line[1].strip() + line[2].strip())
        Y_train.append(int(line[0].strip()))

X_train = np.array(X_train,dtype=object)
Y_train = np.array(Y_train)

X_test = []
Y_test = []

with open('./data/'+test, 'r') as f:
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

##################################
# PROCESS DATA
#################################

print('Get characters...')
alphabet = (list(string.ascii_lowercase) + list(string.digits) + [' '] +
            list(string.punctuation) + ['\n'])
chars = set(alphabet)

vocab_size = len(chars)
print('Vocab size:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

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

##################################
# CNN SETUP
#################################

main_input = Input(shape=(maxlen,vocab_size), name='main_input')

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

out = Dense(Y_train.shape[1], activation='softmax')(z)
model = Model(input=main_input, output=out)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

############################
# RUN MODEL
###########################

history = model.fit(X_train_char,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_test_char, Y_test))
score, acc = model.evaluate(X_test_char, Y_test, batch_size=batch_size)

val_acc = history.history['val_acc']
val_acc_max = np.amax(val_acc)
val_acc_max_ep = np.argmax(val_acc)

print('Best epoch:', val_acc_max_ep+1)
print('Best test error:', 1-val_acc_max)
