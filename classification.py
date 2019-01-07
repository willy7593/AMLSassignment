#import openCV library
import cv2
#import OS module
import os
from cnn3 import CNN_for_tasks
from csv1 import compare_and_get_accuracy

#Main Function
if __name__ == "__main__":

    #these two files are for haar classifier
    haar_face_cascade = cv2.CascadeClassifier('dataset/haarcascade_frontalface_alt.xml')
    haar_eye_cascade = cv2.CascadeClassifier('dataset/haarcascade_eye.xml')

    #initiate count as 0
    count = 0

    #set path
    path = "dataset/dataset/"

    source = os.listdir(path)   #Return a list of the entries in the directory given by path.
    destination1= "training"
    destination2 = "testing"
    
    #This part checks images files and convert them into grayscale and starts to distinguish facial image or not
    # with face and eye classifiers. Next, store them into either training folder or testing folder.
    # 80% of the facial images are used for training.
    for files in source:
        count += 1
        if files.endswith(".png"):  # checking if the files are images in png
            counting = str(count) + '.png'
            img = cv2.imread(os.path.join(path, counting))  # calling images
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert into grayscale

            face = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1,
                                                      minNeighbors=5)  # identify and count faces

            eye = haar_eye_cascade.detectMultiScale(gray_img, scaleFactor=1.1,
                                                    minNeighbors=5)  # identify and count eyes


            if count > 4000:
                if (len(face) > 0)or(len(eye) > 0) :
                    cv2.imwrite(os.path.join(destination2, counting), img)
            else:
                if (len(face) > 0)or(len(eye) > 0):
                    cv2.imwrite(os.path.join(destination1, counting), img)


#from here on is to apply CNN, train,verify and get predictions and accuracy.

#number of epochs
epochs = 1

#Task 1 Emotion recognition
CNN_for_tasks("smiling", "Task1.csv",2,epochs)
compare_and_get_accuracy("smiling", "Task1.csv",3)

#Task 2 Age identification
CNN_for_tasks("young", "Task2.csv",2,epochs)
compare_and_get_accuracy("young", "Task2.csv",4)

#Task 3 Glasses detection
CNN_for_tasks("eyeglasses", "Task3.csv",2,epochs)
compare_and_get_accuracy("eyeglasses", "Task3.csv",2)

#Task 4 Human detection
CNN_for_tasks("human", "Task4.csv",2,epochs)
compare_and_get_accuracy("human", "Task4.csv",5)

#Task 5 Hair colour recognition
CNN_for_tasks("hair_color", "Task5.csv",7,epochs)
compare_and_get_accuracy("hair_color", "Task5.csv",1)
