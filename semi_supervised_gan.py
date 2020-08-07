import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from IPython import display



# select a supervised subset of the dataset, all classes have same number of data points in the subset
def select_samples(X_train, y_train, nb_samples = 1000, nb_classes = 10):
	X_list, y_list = [], []
	nb_per_class = int(nb_samples / nb_classes)
	for class_num in range(nb_classes):
		# get all images for this class
		X_in_class = X_train[y_train == class_num]
		# choose random instances
		rand_instances = np.random.randint(0, len(X_in_class), nb_per_class)
		# add to list
		[X_list.append(X_in_class[j]) for j in rand_instances]
		[y_list.append(class_num) for j in rand_instances]
	return np.asarray(X_list), np.asarray(y_list)

# generate input vector for generator
def generate_input_vector(nb_samples, input_dim = 100):
	norm_input = tf.random.normal([nb_samples, 100])
	return norm_input

#Functional API
def make_generator_model(latent_dim = 100):
	gen_inputs = tf.keras.layers.Input(shape = (latent_dim, ))
	
	dense1 = tf.keras.layers.Dense(128 * 7 * 7)(gen_inputs)
	leaky1 = tf.keras.layers.LeakyReLU(alpha = 0.2)(dense1)
	reshape = tf.keras.layers.Reshape((7, 7, 128))(leaky1)

	convt1 = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides = (2,2), padding = 'same', use_bias = False)(reshape)
	leaky2 = tf.keras.layers.LeakyReLU(alpha = 0.2)(convt1)

	convt2 = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides = (2,2), padding = 'same', use_bias = False)(leaky2)
	leaky3 = tf.keras.layers.LeakyReLU(alpha = 0.2)(convt2)

	g_out_layer = tf.keras.layers.Conv2D(1, (7, 7), padding = 'same', use_bias = False, activation = 'tanh')(leaky3)

	g_model = tf.keras.Model(gen_inputs, g_out_layer)
	return g_model


#Activation to go from softmax preds to logistic
def custom_activation(output):
	logexpsum = tf.keras.backend.sum(tf.keras.backend.exp(output), axis = -1, keepdims = True)
	result = logexpsum / (logexpsum + 1.0)
	return result


#Functional API
def make_discriminator_models(input_shape = (28, 28, 1), n_classes = 10):
	img_inputs = tf.keras.layers.Input(shape = input_shape)
	conv1 = tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same')(img_inputs)
	leaky1 = tf.keras.layers.LeakyReLU(alpha = 0.2)(conv1)

	conv2 = tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same')(leaky1)
	leaky2 = tf.keras.layers.LeakyReLU(alpha = 0.2)(conv2)

	conv3 = tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same')(leaky2)
	leaky3 = tf.keras.layers.LeakyReLU(alpha = 0.2)(conv3)

	conv4 = tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same')(leaky3)
	leaky4 = tf.keras.layers.LeakyReLU(alpha = 0.2)(conv4)

	flat = tf.keras.layers.Flatten()(leaky4)
	drop = tf.keras.layers.Dropout(0.4)(flat)
	dense = tf.keras.layers.Dense(n_classes)(drop)

	s_out_layer = tf.keras.layers.Activation('softmax')(dense)
	s_model = tf.keras.Model(img_inputs, s_out_layer)

	d_out_layer = tf.keras.layers.Lambda(custom_activation)(dense)
	d_model = tf.keras.Model(img_inputs, d_out_layer)

	return s_model, d_model

def sup_loss(y_true, y_pred):
	cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
	return cce_loss(y_true, y_pred)

def discriminator_loss(real_output, fake_output):
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

def generator_loss(fake_output):
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
	return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_sup_step(X, y, s_model, s_opt, class_loss_metric, class_acc_metric):
	with tf.GradientTape() as s_tape:
		sup_preds = s_model(X, training = True)
		supervised_loss = sup_loss(y, sup_preds)

	supervised_grads = s_tape.gradient(supervised_loss, s_model.trainable_variables)
	s_opt.apply_gradients(zip(supervised_grads, s_model.trainable_variables))

	class_loss_metric(supervised_loss)
	class_acc_metric(y, sup_preds)

@tf.function
def train_unsup_step(X_real, half_batch_size, gen_dim, d_model, d_opt, disc_loss_metric, g_model, g_opt, gen_loss_metric):
	X_gen = generate_input_vector(half_batch_size, gen_dim)

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		gen_images = g_model(X_gen, training = True)

		real_output = d_model(X_real, training = True)
		fake_output = d_model(gen_images, training = True)

		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

	gen_grads = gen_tape.gradient(gen_loss, g_model.trainable_variables)
	disc_grads = disc_tape.gradient(disc_loss, d_model.trainable_variables)

	g_opt.apply_gradients(zip(gen_grads, g_model.trainable_variables))
	d_opt.apply_gradients(zip(disc_grads, d_model.trainable_variables))

	disc_loss_metric(disc_loss)
	gen_loss_metric(gen_loss)

@tf.function
def test_step(X, y, s_model, test_loss_metric, test_acc_metric):
	test_preds = s_model(X, training = False)
	test_loss = sup_loss(y, test_preds)

	test_loss_metric(test_loss)
	test_acc_metric(y, test_preds)


def generate_and_save_images(model, epoch, test_input):
	predictions = model(test_input, training=False)

	fig = plt.figure(figsize=(4,4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i + 1)
		plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
		plt.axis('off')

	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
	plt.show()


nb_samples = 1000
nb_classes = 10
batch_size = 100
half_batch_size = batch_size // 2
gen_dim = 100
nb_epochs = 20


#Check to see what the generator is generating
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, gen_dim])


#Loading train and test data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], -1).astype('float32')
X_train = (X_train - 127.5) / 127.5
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype('float32')
X_test = (X_test - 127.5) / 127.5


s_model, d_model = make_discriminator_models()
s_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)
d_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)

g_model = make_generator_model(gen_dim)
g_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)

class_loss_metric = tf.keras.metrics.Mean(name = 'classification_loss')
class_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name = 'classification_accuracy')

disc_loss_metric = tf.keras.metrics.Mean(name = 'discriminator_loss')
gen_loss_metric = tf.keras.metrics.Mean(name = 'generator_loss')

test_loss_metric = tf.keras.metrics.Mean(name = 'test_loss')
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')


#saving models/optimizers
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=g_opt,
	discriminator_optimizer=d_opt,
	generator=g_model,
	discriminator=d_model,
	classifier=s_model)



X_sup, y_sup = select_samples(X_train, y_train, nb_samples, nb_classes)
supervised_ds = tf.data.Dataset.from_tensor_slices((X_sup, y_sup)).shuffle(nb_samples).batch(batch_size)

X_unsup, _ = select_samples(X_train, y_train, nb_samples, nb_classes)
unsupervised_ds = tf.data.Dataset.from_tensor_slices(X_unsup).shuffle(nb_samples).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)



for epoch in range(1, nb_epochs + 1):

	print(epoch)
	start = time.time()

	class_loss_metric.reset_states()
	class_acc_metric.reset_states()
	disc_loss_metric.reset_states()
	gen_loss_metric.reset_states()
	test_loss_metric.reset_states()
	test_acc_metric.reset_states()

	# Train s_model (softmax)
	for sup_images, sup_labels in supervised_ds:
		train_sup_step(sup_images, sup_labels, s_model, s_opt, class_loss_metric, class_acc_metric)

	# Train d_model/g_model (discriminator/generator)
	for unsup_images in unsupervised_ds:
		train_unsup_step(unsup_images, half_batch_size, gen_dim, d_model, d_opt, disc_loss_metric, g_model, g_opt, gen_loss_metric)

	for test_images, test_labels in test_ds:
		test_step(test_images, test_labels, s_model, test_loss_metric, test_acc_metric)

	#Used to examine the quality of the generated images
	display.clear_output(wait = True)
	generate_and_save_images(g_model, epoch + 1, seed)

	if (epoch + 1) % 15 == 0:
		checkpoint.save(file_prefix = checkpoint_prefix)

	print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))



	template = 'Epoch {}, Classification Loss: {}, Classification Accuracy: {}, Discriminator Loss: {}, Generator Loss: {}, Test Loss {}, Test Accuracy: {}'
	print(template.format(epoch,
			class_loss_metric.result(),
			class_acc_metric.result() * 100,
			disc_loss_metric.result(),
			gen_loss_metric.result(),
			test_loss_metric.result(),
			test_acc_metric.result() * 100))

s_model.save('s_model.h5')


#Testing the loading of the saved model and prediction accuracy
# new_model = tf.keras.models.load_model('s_model.h5')

# new_test_preds = s_model(X_test, training = False)
# new_test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# new_test_acc_metric(y_test, new_test_preds)
# print(new_test_acc_metric.result() * 100)



