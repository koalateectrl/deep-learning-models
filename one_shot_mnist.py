import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from absl import flags
from absl import app
from absl import logging

def load_mnist():
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

	X_train = X_train / 255.0
	X_test = X_test / 255.0

	return X_train, y_train, X_test, y_test

def create_data_format(X, y):
	unique, counts = np.unique(y, return_counts=True)
	min_count = np.min(counts)
	X_filtered_set = []
	y_filtered_set = []
	for elem in unique:
		indices = [i for i, item in enumerate(y) if item == elem]
		indices = indices[:min_count]
		unique_tmp = [X[i] for i in indices]
		target_tmp = [y[i] for i in indices]
		X_filtered_set.append(unique_tmp)
		y_filtered_set.extend(target_tmp)
	X_filtered_arr = np.asarray(X_filtered_set, dtype=np.float32)
	y_filtered_arr = np.asarray(y_filtered_set, dtype=np.int8)
	return X_filtered_arr, y_filtered_arr


def get_batch(data_set, batch_size):
    num_classes, num_examples, height, width = data_set.shape
    categories = np.random.choice(num_classes, size = num_classes, replace = False)
    
    #Create pairs(2 images put together) and targets (half are the same character and half are different)
    pairs = np.zeros((2, batch_size, height, width))
    target = np.zeros((batch_size))
    target[batch_size // 2:] = 1
    
    for i in range(batch_size):
        category = categories[i % num_classes]
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
		self.conv1 = tf.keras.layers.Conv2D(64, (2, 2), activation = 'relu')
		self.pool1 = tf.keras.layers.MaxPooling2D()
		self.conv2 = tf.keras.layers.Conv2D(128, (2, 2), activation = 'relu')
		self.pool2 = tf.keras.layers.MaxPooling2D()
		self.conv3 = tf.keras.layers.Conv2D(128, (2, 2), activation = 'relu')
		self.pool3 = tf.keras.layers.MaxPooling2D()
		self.conv4 = tf.keras.layers.Conv2D(256, (2, 2), activation = 'relu')
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
def train_step(images, labels, model_anc, model_pos, model_neg, optimizer, train_loss, train_accuracy):
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


ckpt_path = '/Users/samwwong/Downloads/saved_model/my_model'

X_train, y_train, X_test, y_test = load_mnist()
X_train, y_train = create_data_format(X_train, y_train)
X_test, y_test = create_data_format(X_test, y_test)

batch_size = 32
steps = 1000


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
		model.save(ckpt_path)

	test_step(test_pairs, test_targets, model, test_loss, test_accuracy)

	if step % 100 == 0:
		template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss {}, Test Accuracy: {}'
		print(template.format(step, 
			train_loss.result(),
			train_accuracy.result() * 100,
			test_loss.result(),
			test_accuracy.result() * 100))



# X_small_test_set = []
# for i in range(X_test.shape[0]):
# 	X_tmp = X_test[i][0:3]
# 	X_small_test_set.append(X_tmp)

# X_small_test_arr = np.asarray(X_small_test_set, dtype=np.float32)

# img_1 = X_small_test_arr[0][0]
# img_2 = X_small_test_arr[1][0]
# plt.imshow(img_1)
# plt.show()
# plt.imshow(img_2)
# plt.show()
# img_1 = img_1.reshape(1, img_1.shape[0], img_1.shape[1], -1)
# img_2 = img_2.reshape(1, img_2.shape[0], img_2.shape[1], -1)

# print(model([img_1, img_2], training = False))



# img_1 = X_small_test_arr[0][0]
# img_2 = X_small_test_arr[8][0]
# plt.imshow(img_1)
# plt.show()
# plt.imshow(img_2)
# plt.show()
# img_1 = img_1.reshape(1, img_1.shape[0], img_1.shape[1], -1)
# img_2 = img_2.reshape(1, img_2.shape[0], img_2.shape[1], -1)


# print(model([img_1, img_2], training = False))


# img_1 = X_small_test_arr[2][0]
# img_2 = X_small_test_arr[9][0]
# plt.imshow(img_1)
# plt.show()
# plt.imshow(img_2)
# plt.show()
# img_1 = img_1.reshape(1, img_1.shape[0], img_1.shape[1], -1)
# img_2 = img_2.reshape(1, img_2.shape[0], img_2.shape[1], -1)


# print(model([img_1, img_2], training = False))

# img_1 = X_small_test_arr[3][1]
# img_2 = X_small_test_arr[3][2]
# plt.imshow(img_1)
# plt.show()
# plt.imshow(img_2)
# plt.show()
# img_1 = img_1.reshape(1, img_1.shape[0], img_1.shape[1], -1)
# img_2 = img_2.reshape(1, img_2.shape[0], img_2.shape[1], -1)

# print(model([img_1, img_2], training = False))



# model = tf.keras.models.load_model(ckpt_path)




# img_1 = X_small_test_arr[0][0]
# img_2 = X_small_test_arr[1][0]
# plt.imshow(img_1)
# plt.show()
# plt.imshow(img_2)
# plt.show()
# img_1 = img_1.reshape(1, img_1.shape[0], img_1.shape[1], -1)
# img_2 = img_2.reshape(1, img_2.shape[0], img_2.shape[1], -1)

# print(model([img_1, img_2], training = False))



# img_1 = X_small_test_arr[0][0]
# img_2 = X_small_test_arr[8][0]
# plt.imshow(img_1)
# plt.show()
# plt.imshow(img_2)
# plt.show()
# img_1 = img_1.reshape(1, img_1.shape[0], img_1.shape[1], -1)
# img_2 = img_2.reshape(1, img_2.shape[0], img_2.shape[1], -1)


# print(model([img_1, img_2], training = False))


# img_1 = X_small_test_arr[2][0]
# img_2 = X_small_test_arr[9][0]
# plt.imshow(img_1)
# plt.show()
# plt.imshow(img_2)
# plt.show()
# img_1 = img_1.reshape(1, img_1.shape[0], img_1.shape[1], -1)
# img_2 = img_2.reshape(1, img_2.shape[0], img_2.shape[1], -1)

# print(model([img_1, img_2], training = False))

# img_1 = X_small_test_arr[3][1]
# img_2 = X_small_test_arr[3][2]
# plt.imshow(img_1)
# plt.show()
# plt.imshow(img_2)
# plt.show()
# img_1 = img_1.reshape(1, img_1.shape[0], img_1.shape[1], -1)
# img_2 = img_2.reshape(1, img_2.shape[0], img_2.shape[1], -1)

# print(model([img_1, img_2], training = False))




#TODO: 
#Use triplet loss and try again














