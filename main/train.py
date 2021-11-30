import numpy as np
import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, Input, Activation

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import os

img_size = 28

def load_datasets(path):
    # Load each dataset from their .npy binary files
    x_train = np.load(path + 'x_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'x_test.npy')
    y_test = np.load(path + 'y_test.npy')
    return x_train, y_train, x_test, y_test

def normalize_data(x_train, x_test):
    # Since currently each value is between 0-255 int values for color
    # Normalize it to 0-1 float values
    x_train = x_train / 255
    x_test = x_test / 255
    return x_train, x_test

def generate_model2():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', ))
    model.add(Dense(7, activation='softmax', ))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def save_model(model, name):
    model_json = model.to_json()
    path = "model/" + name
    with open(path + ".json", 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(path + ".h5")

def main():
    x_train, y_train, x_test, y_test = load_datasets('data/dataset/')

    x_train, x_test = normalize_data(x_train, x_test)

    model = generate_model2()

    model.build((None,28,28,3))

    model.summary()

    model.fit(x_train, y_train, epochs=20,
            validation_data=(x_test, y_test))

    print("Model Loss: ", model.evaluate(x_test, y_test))
    print("Model Accuracy: ",
          model.evaluate(x_test, y_test))

    # Save model
    save_model(model, "modelfin")

if __name__ == '__main__':
    main()