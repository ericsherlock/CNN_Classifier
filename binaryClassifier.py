#! /usr/bin/env python

#Import Necessary Packages
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils
from matplotlib import pyplot
import numpy as np

#Read In An Process Text Corpora
#corpora = open("BlogGenderDataset.txt", "r").readlines()
corpora = open("SubjectiveObjectiveDataset.txt", "r").readlines()
corpLab = [1]*(len(corpora)/2)
corpLab.extend([0]*(len(corpora)/2))

#Set Up Tokenizer
tokenizer = Tokenizer(nb_words=10000)
tokenizer.fit_on_texts(corpora)
textSeq = tokenizer.texts_to_sequences(corpora)
wordIndex = tokenizer.word_index

#Pad The Sentences To The Required Length
print('Found %s unique tokens.' % len(wordIndex))
data = pad_sequences(textSeq, maxlen=100)
corpLab = np_utils.to_categorical(np.asarray(corpLab))
print('Shape of Sentence Matrix', data.shape)
print('Shape of Label Matrix', corpLab.shape)

#Randomly Shuffle Data To Split Into Training And Testing
index = np.arange(data.shape[0])
np.random.shuffle(index)
data = data[index]
corpLab = corpLab[index]
corpSplit = int(.1 * data.shape[0])

#Split The Dataset Into Training And Testing
x_train = data[:-corpSplit]
x_test = data[-corpSplit:]
y_train = corpLab[:-corpSplit]
y_test = corpLab[-corpSplit:]

#Check Shapes Of Training and Testing 
print("SHAPE XTRAIN: ", x_train.shape)
print("SHAPE YTRAIN: ", y_train.shape)
print("SHAPE XTEST: ", x_test.shape)
print("SHAPE YTEST: ", y_test.shape)

#Load The Embeddings From Word2Vec File
embedFile = open("GoogleNews-vectors-negative300.bin.gz.txt", "r")
embedIndex = {}
i = 0
for line in embedFile:
    vectors = line.split()
    word = vectors[0]
    wordNumList = np.asarray(vectors[1:], dtype='float32')
    embedIndex[word] = wordNumList
    embedDim = len(wordNumList)
    i = i + 1
    if i > 10000:
        break
embedFile.close()

#Create Embedding Matrix From Embedding Index
embedMatrix = np.zeros((len(wordIndex) + 1, embedDim))
for word, i in wordIndex.items():
    embedVector = embedIndex.get(word)
    if embedVector is not None:
       embedMatrix[i] = embedVector


#Create The Model Object 
model = Sequential()

#Create Input Shape Tensor
shapeTensor = Input(shape=(100,), dtype='int32')
    
#Add Embedding Layer
embedLayer = Embedding(len(wordIndex)+1, 300, input_length=100, weights=[embedMatrix], trainable=False)(shapeTensor)

#Add Convolution Layers
featList = []

#Add Layer, Filter Length 3
convolution = Conv1D(nb_filter=3, filter_length=3, activation='relu')(embedLayer)
poolLayer = MaxPooling1D(pool_length=2)(convolution)
flatPool = Flatten()(poolLayer)
featList.append(flatPool)

#Add Layer, Filter Length 4
convolution = Conv1D(nb_filter=4, filter_length=4, activation='relu')(embedLayer)
poolLayer = MaxPooling1D(pool_length=2)(convolution)
flatPool = Flatten()(poolLayer)
featList.append(flatPool)

#Add Layer, Filter Length 5
convolution = Conv1D(nb_filter=5, filter_length=5, activation='relu')(embedLayer)
poolLayer = MaxPooling1D(pool_length=2)(convolution)
flatPool = Flatten()(poolLayer)
featList.append(flatPool)


#Merge Tensors From Features List Into Single Entity And Add Them To The Model
mergLayers = Concatenate(axis=-1)(featList)
model.add(Model(input=shapeTensor, output=mergLayers))

#Connect Rest Of Model To Dense Layer To Connect To Output
model.add(Dense(100,init='uniform',activation='relu'))
model.add(Dense(2, init='uniform', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train and Test The Model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=50)

#Plot The Results
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('Accuracy/Loss as %')
pyplot.xlabel('Number Epochs')
pyplot.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], loc='upper right')
pyplot.show()
