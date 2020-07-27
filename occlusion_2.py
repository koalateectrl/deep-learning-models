import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def test_step(images, model):
	prediction = model(images)
	return prediction

#Loading in the image
path_to_image = "/Users/samwwong/Downloads/rugby.jpeg"
image = plt.imread(path_to_image)
h, w, c = image.shape

#Preprocessing the image
IMG_SIZE = 224
image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
image = tf.keras.applications.vgg16.preprocess_input(image)
image = tf.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

#Modeling and prediction
vgg_model = tf.keras.applications.VGG16(weights = 'imagenet')
prediction = test_step(image, vgg_model)

#Results
max_idx = tf.math.argmax(prediction, axis = 1)
max_val = prediction[0, max_idx[0]]