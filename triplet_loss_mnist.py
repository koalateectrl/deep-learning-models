import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from absl import flags
from absl import app
from absl import logging

from annoy import AnnoyIndex

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
    assert(num_examples > batch_size * 2)
    categories = np.random.choice(num_classes, size = num_classes, replace = False)
    
    #Create triplets(3 images put together in the order of anchor, positive, and negative)
    triplets = np.zeros((3, batch_size, height, width))
    
    for i in range(batch_size):
        category = categories[i % num_classes]
        anchor_img_index = np.random.randint(0, num_examples)
        pos_img_index = np.random.randint(0, num_examples)
        triplets[0][i][:][:] = data_set[category][anchor_img_index]
        triplets[1][i][:][:] = data_set[category][pos_img_index]
        
        neg_img_index = np.random.randint(0, num_examples)
        category_2 = (category + np.random.randint(1, num_classes)) % num_classes
        
        triplets[2][i][:][:] = data_set[category_2][neg_img_index]
    
    return triplets


class OneTripletModel(tf.keras.Model):
	def __init__(self):
		super(OneTripletModel, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(64, (2, 2), activation = 'relu')
		self.pool1 = tf.keras.layers.MaxPooling2D()
		self.conv2 = tf.keras.layers.Conv2D(128, (2, 2), activation = 'relu')
		self.pool2 = tf.keras.layers.MaxPooling2D()
		self.conv3 = tf.keras.layers.Conv2D(128, (2, 2), activation = 'relu')
		self.pool3 = tf.keras.layers.MaxPooling2D()
		self.conv4 = tf.keras.layers.Conv2D(256, (2, 2), activation = 'relu')
		self.flat = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(4096, activation = 'sigmoid')
		self.dense2 = tf.keras.layers.Dense(50)

	def call(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.pool3(x)
		x = self.conv4(x)
		x = self.flat(x)
		x = self.dense1(x)
		x = self.dense2(x)
		return x

	def model(self, input_shape = (28, 28, 1)):
		x = tf.keras.Input(shape = input_shape)
		return tf.keras.Model(inputs = [x], outputs = self.call(x))


def loss_fn(anc_enc, pos_enc, neg_enc, margin):
	anc_pos_dist = tf.sqrt(tf.reduce_sum(tf.pow(anc_enc - pos_enc, 2), 1, keepdims = True))
	anc_neg_dist = tf.sqrt(tf.reduce_sum(tf.pow(anc_enc - neg_enc, 2), 1, keepdims = True))
	return tf.reduce_mean(tf.maximum(anc_pos_dist - anc_neg_dist + margin, 0))

@tf.function
def train_step(images, anc_model, pos_model, neg_model, anc_optimizer, pos_optimizer, neg_optimizer, train_loss, margin = 0.5):
	with tf.GradientTape() as anc_tape, tf.GradientTape() as pos_tape, tf.GradientTape() as neg_tape:
		anc_enc = anc_model(images[0], training = True)
		pos_enc = pos_model(images[1], training = True)
		neg_enc = neg_model(images[2], training = True)
		loss = loss_fn(anc_enc, pos_enc, neg_enc, margin)

	anc_grads = anc_tape.gradient(loss, anc_model.trainable_variables)
	pos_grads = pos_tape.gradient(loss, pos_model.trainable_variables)
	neg_grads = neg_tape.gradient(loss, neg_model.trainable_variables)

	anc_optimizer.apply_gradients(zip(anc_grads, anc_model.trainable_variables))
	pos_optimizer.apply_gradients(zip(pos_grads, pos_model.trainable_variables))
	neg_optimizer.apply_gradients(zip(neg_grads, neg_model.trainable_variables))

	train_loss(loss)

anc_model = OneTripletModel()
pos_model = OneTripletModel()
neg_model = OneTripletModel()
# print(model.model().summary())


# ckpt_path = '/Users/samwwong/Downloads/saved_model/my_model'

X_train, y_train, X_test, y_test = load_mnist()
X_train, y_train = create_data_format(X_train, y_train)
X_test, y_test = create_data_format(X_test, y_test)


batch_size = 32
triplets = get_batch(X_train, batch_size)
steps = 12


anc_optimizer = tf.keras.optimizers.Adam(lr = 0.0006)
pos_optimizer = tf.keras.optimizers.Adam(lr = 0.0006)
neg_optimizer = tf.keras.optimizers.Adam(lr = 0.0006)

train_loss = tf.keras.metrics.Mean(name = 'train_loss')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')


for step in range(steps):

	train_loss.reset_states()
	test_loss.reset_states()

	triplets = get_batch(X_train, batch_size)

	triplets = triplets.reshape(triplets.shape[0], triplets.shape[1], 
		triplets.shape[2], triplets.shape[3], -1).astype('float32')

	start_time = time.time()

	train_step(triplets, anc_model, pos_model, neg_model, anc_optimizer, pos_optimizer, neg_optimizer, train_loss)

	if step % 100 == 0:
		end_time = time.time()
		logging.info(f"Step {step} time in seconds: {end_time - start_time}")
# 		model.save(ckpt_path)
	
	if step % 100 == 0:
		template = 'Step {}, Loss: {}'
		print(template.format(step, 
			train_loss.result()))


#TESTING USE ANY MODEL


test_images = np.asarray([X_train[i][0][:][:] for i in range(len(X_train))], dtype = np.float32)
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
test_encodings = anc_model.predict(test_images)

X_train = X_train.reshape(-1, X_train.shape[2], X_train.shape[3], 1)

encodings = anc_model.predict(X_train)

num_features = test_encodings.shape[1]  # Length of item vector that will be indexed
t = AnnoyIndex(num_features, 'angular') 

for i in range(test_encodings.shape[0]):
	t.add_item(i, test_encodings[i])

for i in range(encodings.shape[0]):
    t.add_item(i, encodings[i])

t.build(10)

def get_nearest_neighbors(index, num_results = 10):
	v = t.get_item_vector(index)
	return t.get_nns_by_vector(v, num_results)

nearest_neighbors_list = []
for i in range(test_encodings.shape[0]):
	nearest_neighbors_list.append(get_nearest_neighbors(i))


w = 10
h = 10
fig = plt.figure(figsize=(8, 8))
columns = 10
rows = 10
for i in range(1, rows + 1):
	for j in range(1, columns + 1):
		idx = nearest_neighbors_list[i - 1][j - 1]
		img = X_train[idx]
		img = img.reshape(img.shape[0], img.shape[1])
		fig.add_subplot(rows, columns, (i - 1) * rows + j)
		plt.imshow(img)
plt.show()

