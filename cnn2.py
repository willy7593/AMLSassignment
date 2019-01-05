import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import sparse_categorical_crossentropy
from keras.layers.core import Activation

df=pd.read_csv(r"dataset/attribute_list.csv", skiprows=1)


datagen=ImageDataGenerator(rescale=1./255)
train_generator=datagen.flow_from_dataframe(dataframe=df,
                                            directory="training",
                                            x_col="file_name", y_col="smiling",
                                            class_mode="categorical",
                                            subset = "training",
                                            has_ext=False,
                                            target_size=(32,32), batch_size=32)


valid_generator=datagen.flow_from_dataframe(dataframe=df,
                                            directory="testing/",
                                            x_col="file_name", y_col="smiling",
                                            has_ext=False,
                                            class_mode="categorical",
                                            target_size=(32,32), batch_size=32)


model = Sequential()
model.add(Convolution2D(32, (3, 3), padding='same',
                        input_shape=(255,255,3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizers.rmsprop(lr=0.0001,
loss="categorical_crossentropy", metrics=["accuracy"]))

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10)