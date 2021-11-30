import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf

def main():
    dataset = pd.read_csv("data/hmnist_28_28_RGB.csv")

    label = dataset["label"]
    image_data = dataset.drop(columns=["label"])
    print(image_data, label)
    x_train , x_test , y_train , y_test = \
        train_test_split(image_data, label , test_size = 0.2)

    sampler = RandomOverSampler()
    x_train, y_train = sampler.fit_resample(x_train, y_train)


    x_train = np.array(x_train).reshape(-1,28,28,3)
    x_test = np.array(x_test).reshape(-1,28,28,3)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    
    x_val = []
    y_val = []

    for i in range(10):
        r = random.randint(0, len(x_test) - 1)
        x_val.append(x_test[r])
        y_val.append(y_test[r])
    
    x_val = np.array(x_val).reshape(-1,28,28,3)
    y_val = tf.keras.utils.to_categorical(y_val)

    np.save('data/dataset/x_train.npy', x_train)
    np.save('data/dataset/y_train.npy', y_train)
    np.save('data/dataset/x_test.npy', x_test)
    np.save('data/dataset/y_test.npy', y_test)
    np.save('data/dataset/x_val.npy', x_val)
    np.save('data/dataset/y_val.npy', y_val)

if __name__ == '__main__':
    main()
