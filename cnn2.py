import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras import optimizers
import numpy as np


df=pd.read_csv(r"dataset/attribute_list.csv", skiprows=1)


datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)

train_generator=datagen.flow_from_dataframe(dataframe=df,
                                            directory="training",
                                            x_col="file_name", y_col="smiling",
                                            class_mode="categorical",
                                            has_ext=False,
                                            shuffle=True,
                                            seed=42,
                                            subset='training',
                                            target_size=(32,32), batch_size=32)


valid_generator=datagen.flow_from_dataframe(dataframe=df,
                                            directory="training",
                                            x_col="file_name", y_col="smiling",
                                            has_ext=False,
                                            class_mode="categorical",
                                            shuffle=True,
                                            seed=42,
                                            subset='validation',
                                            target_size=(32,32), batch_size=32)

test_generator=datagen.flow_from_dataframe(dataframe=df,
                                            directory="testing",
                                            x_col="file_name", y_col="smiling",
                                            has_ext=False,
                                            class_mode="categorical",
                                            shuffle=True,
                                            seed=42,
                                            target_size=(32,32), batch_size=32)



model = Sequential()
#step 1 - convolution
model.add(Convolution2D(32,3,3, input_shape = (32,32,3), activation ='relu'))

#step 2 - pooling
model.add(MaxPooling2D(pool_size = (2,2)))

#step 3 - Flattening
model.add(Flatten())

#step 4 - Full Connection
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 2, activation = 'sigmoid'))

#compile
model.compile(optimizers.rmsprop(lr=0.001), loss ="categorical_crossentropy", metrics=["accuracy"])

#Training the model through fit
# steps_per_epoch should be (number of training images total / batch_size)
# validation_steps should be (number of validation images total / batch_size)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1
)



#predicting the output
test_generator.reset()
pred=model.predict_generator(test_generator,steps= len(test_generator),verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


#Finally, save the results to a CSV file.
filenames=test_generator.filenames
results=pd.DataFrame({"file_name":filenames,
                      "Predictions":predictions})
results.to_csv("Task1.csv",index=False)

#removing the .png extension on task1 file_name
text = open("Task1.csv", "r")
text = ''.join([i for i in text]).replace(".png", "")
x = open("Task1.csv","w")
x.writelines(text)
x.close()

