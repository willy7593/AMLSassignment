# import the necessary packages
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Activation, Dropout
from keras.layers.core import Flatten, Dense
from keras.optimizers import Adam
from keras.metrics import sparse_categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils




from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense

#Initialize the CNN
classifier = Sequential()

#step 1 - convolution
classifier.add(Convolution2D(32,3,3, input_shape = (255,255,3), activation ='relu'))

#step 2 - pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step 3 - Flattening
classifier.add(Flatten())

#step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activations= 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary crossentropy', metrics=['accuracy'])



train_path = "training/"
test_path = "testing/"
valid_path = "valid/"

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),classes=['smile','no_smile'], batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224),classes=['smile','no_smile'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224),classes=['smile','no_smile'], batch_size=10)

#check image shape
print (X_train[0].shape)


#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)



#create model
model = Sequential()
#add model layers, first layer is 2D convolutionary layer 
#channel is 3 for rgb instead of 1 for grayscale
model.add(Conv2D(64, kernel_size=3, activation='reLu', input_shape=(255,255,3)))
model.add(Conv2D(32, kernel_size=3, activation='reLu'))
model.add(Flatten())
model.add(Dense(128, activation='softmax'))
model.add(Dense(2, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit_generator(train_batches, steps_per_epoch = 4, validation_data= valid_batches, validation_steps=4, epochs=5, verbose=2)

#predict

#test data and image for prediction
#predict generator (keras library)
#change array into a list (structure)


#compare test's label and give accuracy
#take only the relevant info where the unrelevant pictures and other labels are deleted. Then compare the two lists.

#finally, export into csv as name file and prediction and on top is accuracy (use panda toCSV)
