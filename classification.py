
# import lab2_landmarks as l2
#import open cv library
import cv2
#import matplotlib library
import matplotlib.pyplot as plt
import os
import numpy as np
#shutil is the file managing library
import shutil

#kera is used for ML
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten





def load_images_from_folder():
    # Open a file
    path = "dataset/dataset/"

    source = os.listdir(path)
    destination = "dataset/output/"
    for files in source:
        if files.endswith(".png"):
            img = cv2.imread(os.path.join(path, files))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5);
            if face.any() > 0:
                shutil.copy(files, destination)
    return 0

if __name__ == "__main__":

    haar_face_cascade = cv2.CascadeClassifier('dataset/haarcascade_frontalface_alt.xml')
    haar_eye_cascade = cv2.CascadeClassifier('dataset/haarcascade_eye.xml')

    count = 0
    path = "dataset/dataset/"

    source = os.listdir(path)
    destination1= "training"
    destination2 = "testing"
    for files in source:
        count += 1
        if files.endswith(".png"):
            counting = str(count) + '.png'
            img = cv2.imread(os.path.join(path, counting))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5);

            if count > 4000:
                if len(face) > 0:
                    cv2.imwrite(os.path.join(destination2, counting), img)
            else:
                if len(face) > 0:
                    cv2.imwrite(os.path.join(destination1, counting), img)

##I could have added an extra thing about nose
##I could also have a result for LBP
##lbp_face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')
##lbp_detected_img = detect_faces(lbp_face_cascade, test1)

##need to do a valid generator for the 1/4 of training set



















'''
def get_data():
    X, y = l2.extract_features_labels()
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:100]
    tr_Y = Y[:100]
    te_X = X[100:]
    te_Y = Y[100:]
    return tr_X, tr_Y, te_X, te_Y


def train_SVM(training_images, training_labels, test_images, test_labels):

    return 0


def train_MLP(training_images, training_labels, test_images, test_labels):

    return 0

if __name__ == "__main__":
    tr_X, tr_Y, te_X, te_Y = get_data()
    '''