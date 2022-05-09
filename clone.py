import sys
import csv
from tqdm import tqdm
import cv2
import datetime
import h5py
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------
# add for data augument
import pandas as pd
import random
import ntpath
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa


def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image

def pan(image):
    pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

def img_random_flip(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return image, steering_angle

def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = img_random_flip(image, steering_angle)

    return image, steering_angle


def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


def batch_generator(image_paths, steering_ang, batch_size, istraining):

    while True:
        batch_img = []
        batch_steering = []

        for i in range(batch_size):
            
            random_index = random.randint(0, len(image_paths) - 1)
           
            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]
            
            im = img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))


# ----------------------------------------------------------------


def getrowsFromDrivingLogs(dataPath):
    rows = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)            
        for row in reader:
            rows.append(row)
    return rows

def getImageArray3angle(imagePath, steering, images, steerings):
    originalImage = cv2.imread(imagePath.strip())
    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
    images.append(image)
    steerings.append(steering)
    
def getImagesAndSteerings(rows):
    
    images = []
    steerings = []
    
    for row in tqdm(rows):
        #angle
        steering = float(row[3])
        #center
        getImageArray3angle(row[0], steering, images, steerings)
        #left

        #right
        
    
    return (np.array(images), np.array(steerings))


# def trainModelAndSave(model, inputs, outputs, epochs, batch_size):
    
#     X_train, X_valid, y_train, y_valid = train_test_split(inputs, outputs, test_size=0.2, shuffle=True)
#     #Setting model
#     model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-3))
#     #Learning model
#     model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(X_valid, y_valid))
#     #Saving model
#     model.save('model.h5')

    
def trainModelAndSave(model, inputs, outputs, epochs, batch_size):
    
    X_train, X_valid, y_train, y_valid = train_test_split(inputs, outputs, test_size=0.2, shuffle=True)
    #Setting model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-3))
    #Learning model
    #model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(X_valid, y_valid))
    
    history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                              steps_per_epoch=300,
                              epochs=10,
                              validation_data=batch_generator(X_valid, y_valid, 100, 0),
                              validation_steps=200,
                              verbose=1,
                              shuffle = 1)
    
    #Saving model
    model.save('model.h5')



#essential network
def nn_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    return model

#convolutional network
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, 3, 3, subsample=(2,2), activation='relu', input_shape=(160, 320, 3)))
    model.add(Conv2D(64, 3, 3, subsample=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    return model

#NVIDIA
def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Conv2D(24,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(36,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(48,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(64,3,3, activation='elu'))
    model.add(Conv2D(64,3,3, activation='elu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    model.add(Dense(100, activation='elu'))
#    model.add(Dropout(0.5))

    model.add(Dense(50, activation='elu'))
#    model.add(Dropout(0.5))

    model.add(Dense(10, activation='elu'))
#    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    
    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model

if __name__ == "__main__":
    
    epochs = 10
    #Make sure to set batch size within 40
    batch_size = 40
    is_dataset = True
    #is_dataset = False
    
    #When making "is_dataset" True, saving preprocessed datasets
    #When making "is_dataset" False, using preprocessed and saved datasets. Shortens time because preprocessing is not required.It must be preprocessed once.
    if is_dataset:
        print('get csv data from Drivinglog.csv')
        rows = getrowsFromDrivingLogs('data')
        print('preprocessing data...')
        inputs, outputs = getImagesAndSteerings(rows)
        
        with h5py.File('./trainingData.h5', 'w') as f:
            f.create_dataset('inputs', data=inputs)
            f.create_dataset('outputs', data=outputs)
    
    else:
        with h5py.File('./trainingData.h5', 'r') as f:
            inputs = np.array(f['inputs'])
            outputs = np.array(f['outputs'])

    print('Training data:', inputs.shape)
    print('Training label:', outputs.shape)
    
    #Specifying model
    #model = nvidia_model()
    model = nvidia_model()
    #Training and saving model
    trainModelAndSave(model, inputs, outputs, epochs, batch_size)
    model.save('model.h5')