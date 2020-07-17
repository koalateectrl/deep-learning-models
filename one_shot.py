import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from absl import flags
from absl import app
from absl import logging

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

#Can use this if you want to load data in
def load_images(path, n = 0):
	X = []
	y = []
	cat_dict = {}
	lang_dict = {}
	curr_y = n

	for alphabet in listdir_nohidden(path):
		print("loading alphabet: " + alphabet)
		lang_dict[alphabet] = [curr_y, None]
		alphabet_path = os.path.join(path, alphabet)

		for letter in listdir_nohidden(alphabet_path):
			cat_dict[curr_y] = (alphabet, letter)
			category_images = []
			letter_path = os.path.join(alphabet_path, letter)

			for filename in listdir_nohidden(letter_path):
				image_path = os.path.join(letter_path, filename)
				image = plt.imread(image_path)
				category_images.append(image)
				y.append(curr_y)
			try:
				X.append(np.stack(category_images))
			except ValueError as e:
				print(e)
				print("error - category_images:", category_images)
			curr_y += 1
			lang_dict[alphabet][1] = curr_y - 1
	y = np.vstack(y)
	X = np.stack(X)
	return X, y, lang_dict

def load_mnist():
	(X_train, y_train), (X_test, y_test) = tf.keras.Datasets.mnist.load_data()


def get_batch(data_set, batch_size):
    num_classes, num_examples, height, width = data_set.shape
    categories = np.random.choice(num_classes, size = batch_size, replace = False)
    
    #Create pairs(2 images put together) and targets (half are the same character and half are different)
    pairs = np.zeros((2, batch_size, height, width))
    target = np.zeros((batch_size))
    target[batch_size // 2:] = 1
    
    for i in range(batch_size):
        category = categories[i]
        first_img_index = np.random.randint(0, num_examples)
        pairs[0][i][:][:] = data_set[category][first_img_index]
        
        second_img_index = np.random.randint(0, num_examples)
        #The first half of the vector is for images of the same character, the second half is for different
        if i >= batch_size // 2:
            category_2 = category
        else:
            category_2 = (category + np.random.randint(1, num_classes)) % num_classes
        
        pairs[1][i][:][:] = data_set[category_2][second_img_index]
    
    return pairs, target


class SiameseModel(tf.keras.Model):
	def __init__(self):
		super(SiameseModel, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(64, (10, 10), activation = 'relu')
		self.pool1 = tf.keras.layers.MaxPooling2D()
		self.conv2 = tf.keras.layers.Conv2D(128, (7, 7), activation = 'relu')
		self.pool2 = tf.keras.layers.MaxPooling2D()
		self.conv3 = tf.keras.layers.Conv2D(128, (4, 4), activation = 'relu')
		self.pool3 = tf.keras.layers.MaxPooling2D()
		self.conv4 = tf.keras.layers.Conv2D(256, (4, 4), activation = 'relu')
		self.flat = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(4096, activation = 'sigmoid')
		self.diff = tf.keras.layers.Lambda(lambda x: tf.keras.backend.abs(x[0] - x[1]))
		self.dense2 = tf.keras.layers.Dense(1, activation = 'sigmoid')

	def call(self, x):
		first_input = x[0]
		second_input = x[1]

		f = self.conv1(first_input)
		f = self.pool1(f)
		f = self.conv2(f)
		f = self.pool2(f)
		f = self.conv3(f)
		f = self.pool3(f)
		f = self.conv4(f)
		f = self.flat(f)
		f = self.dense1(f)

		s = self.conv1(second_input)
		s = self.pool1(s)
		s = self.conv2(s)
		s = self.pool2(s)
		s = self.conv3(s)
		s = self.pool3(s)
		s = self.conv4(s)
		s = self.flat(s)
		s = self.dense1(s)

		d = self.diff([f, s])
		d = self.dense2(d)
		return d

	def model(self, input_shape = (28, 28, 1)):
		x1 = tf.keras.Input(shape = input_shape)
		x2 = tf.keras.Input(shape = input_shape)
		return tf.keras.Model(inputs = [x1, x2], outputs = self.call([x1,x2]))


def loss_fn(y_true, y_pred):
	loss = tf.keras.losses.BinaryCrossentropy()
	return loss(y_true, y_pred)

@tf.function
def train_step(images, labels, model, optimizer, train_loss, train_accuracy):
	with tf.GradientTape() as tape:
		predictions = model([images[0], images[1]], training = True)
		loss = loss_fn(labels, predictions)

	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))

	train_loss(loss)
	train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels, model, test_loss, test_accuracy):
	predictions = model([images[0], images[1]], training = False)
	loss = loss_fn(labels, predictions)

	test_loss(loss)
	test_accuracy(labels, predictions)


model = SiameseModel()
# print(model.model().summary())


train_folder_path = '/Users/samwwong/Downloads/images_background'
test_folder_path = '/Users/samwwong/Downloads/images_evaluation'
ckpt_path = '/Users/samwwong/Downloads/'

print("Loading Train Classes!")
X_train, y_train, train_class_dict = load_images(train_folder_path)

print("Loading Test Classes!")
X_test, y_test, test_class_dict = load_images(test_folder_path)

batch_size = 32
steps = 100


optimizer = tf.keras.optimizers.Adam(lr = 0.0006)

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name = 'test_accuracy')


for step in range(steps):

	train_loss.reset_states()
	train_accuracy.reset_states()
	test_loss.reset_states()
	test_accuracy.reset_states()

	train_pairs, train_targets = get_batch(X_train, batch_size)
	test_pairs, test_targets = get_batch(X_test, batch_size)

	train_pairs = train_pairs.reshape(train_pairs.shape[0], train_pairs.shape[1], 
		train_pairs.shape[2], train_pairs.shape[3], -1).astype('float32')

	test_pairs = test_pairs.reshape(test_pairs.shape[0], test_pairs.shape[1], 
		test_pairs.shape[2], test_pairs.shape[3], -1).astype('float32')

	start_time = time.time()

	train_step(train_pairs, train_targets, model, optimizer, train_loss, train_accuracy)

	if step % 100 == 0:
		end_time = time.time()
		logging.info(f"Step {step} time in seconds: {end_time - start_time}")

	model.save_weights(ckpt_path)

	test_step(test_pairs, test_targets, model, test_loss, test_accuracy)


	template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss {}, Test Accuracy: {}'
	print(template.format(step, 
		train_loss.result(),
		train_accuracy.result() * 100,
		test_loss.result(),
		test_accuracy.result() * 100))


#TODO: check - does the model actually learn things?
#Implemenet on MNIST or FASHION MNIST?
#How come when we have small size input the network doesn't work?

#Upgrade to 3 
#Use triplet loss and try again














