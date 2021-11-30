import json
import sys
import random
from csv import reader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.core.fromnumeric import argmax
from tensorflow.keras.models import Sequential, model_from_json, load_model

img_size = 28
labels = {
    # Not cancer
    0:'Actinic Keratoses', 
    # Cancer
    1:'Basal Cell Carcinoma',
    # Not cancer
    2:'Benign Keratosis-like Lesions',
    # Not cancer
    3:'Dermatofibroma',
    # Usually noncancerous
    4:'Melanocytic Nevi',
    # Not cancer
    5:'Vascular Lesions',
    # Cancer
    6:'Melanoma'
}

shorthand = {
    'akiec':'Actinic Keratoses',
    'bcc':'Basal Cell Carcinoma',
    'bkl':'Benign Keratosis-like Lesions',
    'df':'Dermatofibroma',
    'nv':'Melanocytic Nevi',
    'vasc':'Vascular Lesions',
    'mel':'Melanoma'
}

def load_model():
    with open("model/modelfin.json", 'r') as json_file:
        json_model = json_file.read()

        model = model_from_json(json_model)

        model.load_weights("model/modelfin.h5")
        print("Loaded Model")
        return model

def displayImage(img, guess, confidence, actual):
    plt.style.use("dark_background")
    fig = plt.figure()
    plt.imshow(img)
    title = 'Prediction ' + guess + '\nConfidence ' + confidence
    if actual:
        title += '%\n Actual ' + actual

    plt.title(title, size=10)
    plt.show()

def formatImage(model, path):
    img = Image.open(path, 'r')
    resized_image = img.resize((28, 28))

    imgarray = np.array(resized_image)
    imgarray = np.array(resized_image) / 255
    imgarray = imgarray.reshape(-1, img_size, img_size, 3)

    return img, imgarray

def main(args):
    model = load_model()

    if len(args) > 1:
        path = args[1]
        actual = None
        if len(args) > 2:
            actual = args[2]
        img, imgarray = formatImage(model, path)
        predictions = model.predict(imgarray)[0]
        guess = predictions.argmax()
        displayImage(img, str(labels[guess]), str(round(100 * predictions[guess], 2)), actual)
        return

    metadata = []
    with open('data/HAM10000_metadata.csv') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            metadata += [row]

    i = 0
    while i < 20:
        imagenum = random.randint(0, len(metadata) - 1)
        path = 'data/test/' + metadata[imagenum][1] + '.jpg'
        actual = metadata[imagenum][2]
        try:
            img, imgarray = formatImage(model, path)
        except:
            print('error')
            continue
        predictions = model.predict(imgarray)[0]
        guess = predictions.argmax()
        displayImage(img, str(labels[guess]), str(round(100 * predictions[guess], 2)), actual)
        i += 1

if __name__ == '__main__':
    main(sys.argv)