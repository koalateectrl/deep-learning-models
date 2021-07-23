import tensorflow as tf
import numpy as np
import time

from absl import flags
from absl import app
from absl import logging

flags.DEFINE_boolean(
    'dpsgd', False, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 5, 'Number of epochs')
#***What does this do?
flags.DEFINE_integer(
    'microbatches', 256, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS


def create_data_set():
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
	X_train = X_train / 255.0
	X_test = X_test / 255.0

	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], -1).astype("float32")
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype("float32")

	train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(60000).batch(FLAGS.batch_size)
	test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(FLAGS.batch_size)
	return train_ds, test_ds


#model subclassing API
class MyModel(tf.keras.Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(16, 8, strides = 2, padding = 'same', activation = 'relu')
		self.pool1 = tf.keras.layers.MaxPool2D(2, 1)
		self.conv2 = tf.keras.layers.Conv2D(32, 4, strides = 2, padding = 'valid', activation = 'relu')
		self.pool2 = tf.keras.layers.MaxPool2D(2, 1)
		self.flat = tf.keras.layers.Flatten()
		self.d1 = tf.keras.layers.Dense(32, activation = 'relu')
		self.d2 = tf.keras.layers.Dense(10)

	def call(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.flat(x)
		x = self.d1(x)
		x = self.d2(x)
		return x


def loss_fn(y_true, y_pred):
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
	# vector_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
	return loss(y_true, y_pred)

@tf.function
def train_step(images, labels, model, optimizer, train_loss, train_accuracy):
	with tf.GradientTape() as tape:
		predictions = model(images, training = True)
		loss = loss_fn(labels, predictions)
		# print(loss)
		# print(loss.shape)

	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	train_loss(loss)
	train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels, model, test_loss, test_accuracy):
	predictions = model(images, training = False)
	loss = loss_fn(labels, predictions)
	# scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

	test_loss(loss)
	test_accuracy(labels, predictions)


def main(unused_argv):
	logging.set_verbosity(logging.INFO)
	train_ds, test_ds = create_data_set()
	#Instantiate the Model.
	model = MyModel()
	optimizer = tf.keras.optimizers.SGD(FLAGS.learning_rate)

	train_loss = tf.keras.metrics.Mean(name = 'train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

	test_loss = tf.keras.metrics.Mean(name = 'test_loss')
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

	#Training loop.
	#****It appears they shuffle the training set each time they select a mini-batch. We go through the list sequentially
	#****I think they are picking a random group from the training set as val. We will do the same.
	for epoch in range(1, FLAGS.epochs + 1):
		start_time = time.time()
		#Train the model for one epoch.
		train_loss.reset_states()
		train_accuracy.reset_states()
		test_loss.reset_states()
		test_accuracy.reset_states()
		# steps = 0

		print(train_ds.shape)

		for images, labels in train_ds:
			# print(steps)
			# steps += 1
			train_step(images, labels, model, optimizer, train_loss, train_accuracy)

		end_time = time.time()
		logging.info(f"Epoch {epoch} time in seconds: {end_time - start_time}")

		for test_images, test_labels in test_ds:
			test_step(test_images, test_labels, model, test_loss, test_accuracy)

		template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss {}, Test Accuracy: {}'
		print(template.format(epoch,
			train_loss.result(),
			train_accuracy.result() * 100,
			test_loss.result(),
			test_accuracy.result() * 100))

if __name__ == '__main__':
	app.run(main)







