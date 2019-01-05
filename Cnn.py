from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.optimizers import Adam
from keras.metrics import sparse_categorical_crossentropy


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
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#part 2 - fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale =1./255)

training_set = train_datagen.flow_from_directory('training/',
                                                 target_size=(255,255),
                                                 batch_size = 2,
                                                 class_mode= 'binary',
                                                 classes=['smile', 'no_smile'])

test_set = test_datagen.flow_from_directory('testing/',
                                            target_size=(255,255),
                                            batch_size = 2,
                                            class_mode= 'binary',
                                            classes=['smile', 'no_smile'])


from IPython.display import display


classifier.fit_generator(training_set,
                         nb_epoch=10,
                         samples_per_epoch = 2000,
                         validation_data= test_set,
                         nb_val_samples=40)





