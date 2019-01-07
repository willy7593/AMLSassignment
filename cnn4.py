#kera is used for ML
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras import optimizers
import numpy as np


#This Function generates CNN model and creates a prediction for the model through taking in the images and trained
# with attribute_list.csv. We can choose the task and the number of epochs where shape is the number of attributes,
# eg. smiling or not smiling will have a shape of two.
def CNN_for_tasks(task, number_of_task,shape,epochs):

    #read the attribute_list.csv
    df = pd.read_csv(r"dataset/attribute_list.csv", skiprows=1)
    #initiate ImageDataGenerator while splitting validation to 25% of the training images
    datagen = ImageDataGenerator(validation_split=0.25, rescale=1. / 255)

    #Initializing train_generator using flow_from_dataframe method. I've turned on shuffle for more consistent result
    # using seed 42. the batch_size is 10 and target size is 32x32 pixel
    train_generator = datagen.flow_from_dataframe(dataframe=df,
                                                  directory="training",
                                                  x_col="file_name", y_col=task,
                                                  class_mode="categorical",
                                                  has_ext=False,
                                                  shuffle=True,
                                                  seed = 3,
                                                  subset='training',
                                                  target_size=(32, 32), batch_size=10)

    valid_generator = datagen.flow_from_dataframe(dataframe=df,
                                                  directory="training",
                                                  x_col="file_name", y_col=task,
                                                  has_ext=False,
                                                  class_mode="categorical",
                                                  shuffle=True,
                                                  seed = 3,
                                                  subset='validation',
                                                  target_size=(32, 32), batch_size=10)

    test_generator = datagen.flow_from_dataframe(dataframe=df,
                                                 directory="testing",
                                                 x_col="file_name", y_col=task,
                                                 has_ext=False,
                                                 class_mode="categorical",
                                                 shuffle=True,
                                                 seed =3,
                                                 target_size=(32, 32), batch_size=10)

    # Initialising CNN Model
    model = Sequential()
    # adding convolutional layer
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu'))
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu'))
    # pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # dropout
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, input_shape=(32, 32, 3), activation='relu'))
    model.add(Convolution2D(64, 3, 3, input_shape=(32, 32, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(shape, activation='softmax'))

    # compile
    model.compile(optimizers.rmsprop(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

    # Training the model through fit
    # steps_per_epoch should be (number of training images total / batch_size)
    # validation_steps should be (number of validation images total / batch_size)
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=epochs
                        )

    # predicting the output
    test_generator.reset()      #reset test_generator first
    pred = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)
    #generating labels for predictions
    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]


    # Finally, save the results to a CSV file.
    filenames = test_generator.filenames
    results = pd.DataFrame({"file_name": filenames,
                            "Predictions": predictions})
    results.to_csv(number_of_task, index=False)

    # removing the .png extension on task1 file_name
    text = open(number_of_task, "r")
    text = ''.join([i for i in text]).replace(".png", "")
    x = open(number_of_task, "w")
    x.writelines(text)
    x.close()

    return 0
