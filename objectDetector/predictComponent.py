import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


from keras import backend as K

import glob

### ------- Retrieve saved model ----- ###
    # Define the neural network
    # Prepares a model of a Neural Network that we will fit data to
image_size = (28, 28)
input_size = (28,28,1)
batch_size = 32
num_classes = 4

loadedModel = tf.keras.models.load_model('trained_model_4_elements.h5')
loadedModel.summary()

# A few random samples
samples_to_predict = []
filepaths =  glob.glob('test2/*')
num_imgs = len(filepaths)
for filepath in filepaths:
    print(filepath)
    img = load_img(filepath, color_mode = "grayscale")
    img_array = img_to_array(img)
    print(img_array.shape)

    img_resize = tf.image.resize(img_array, image_size, preserve_aspect_ratio=True)
    # predictions_one = loadedModel.predict(samples_to_predict)
    samples_to_predict = np.reshape(img_resize,(1,28,28,1))
    # samples_to_predict.append(img_resize)
# print(samples_to_predict)


# np.empty(shape=(1,28,28,1))
# print(np.ndarray(shape=(1,1,2,3)))

# Convert into Numpy array
samples_to_predict = np.array(samples_to_predict)
# print(samples_to_predict.shape)

# Generate predictions for samples
predictions = loadedModel.predict(samples_to_predict)
# print(predictions)

confidence_scores =np.array(predictions) * (100*np.ones((num_imgs,1)))
# confidence_scores = np.reshape(confidence_scores,(num_classes,1))
# print(confidence_scores)


# Generate arg maxes for predictions
classes = np.argmax(predictions, axis = 1)
# print(classes)

class_names = np.array( ['diodes','resistor','inductor','capacitor'])
# print(class_names.shape)

prediction_names = []
for i in classes:
    prediction_names.append(class_names[i])

# print(prediction_names)

# Create dictionary 
test = dict(zip([1, 2, 3, 4], ['a', 'b', 'c', 'd']))
index = confidence_scores[0,:]
combined_results = dict(zip(index,class_names))
sorted_results = sorted(combined_results.items(), key=lambda kv: kv[0],reverse=True)
# print(combined_results)
print(sorted_results)
# what is it
# orientation or split in half
# confidence 