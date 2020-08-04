import tensorflow as tf
import numpy as np



# select a supervised subset of the dataset, all classes have same number of data points in the subset
def select_supervised_samples(X_train, y_train, nb_samples = 100, nb_classes = 10):
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

# select real samples
def generate_real_samples(X_train, y_train, nb_samples):
	# choose random instances
	rand_instances = np.random.randint(0, X_train.shape[0], nb_samples)
	# select images and labels
	new_X, classes = X_train[rand_instances], y_train[rand_instances]

	is_real = np.ones((nb_samples, 1))
	return [new_X, classes], is_real

def generate_input_vector(nb_samples, input_dim = 100):
	norm_input = np.random.randn(input_dim * nb_samples)
	norm_input = norm_input.reshape(nb_samples, input_dim)
	return norm_input

# create fake samples
def generate_fake_samples(g_model, nb_samples, input_dim = 100):
	input_vector = generate_input_vector(nb_samples, input_dim)
	g_images = g_model(input_vector)
	is_real = np.zeros((nb_samples, 1))
	return g_images, is_real

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

def unsup_loss(y_true, y_pred):
	sigmoid_loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)
	return sigmoid_loss(y_true, y_pred)

def gen_loss(y_true, y_pred):
	sigmoid_loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)
	return sigmoid_loss(y_true, y_pred)


def discriminator_loss(real_output, fake_output):
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

def generator_loss(fake_output):
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
	return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_sup_step(X, y, s_model, s_opt, class_loss, class_acc):
	with tf.GradientTape() as s_tape:
		sup_preds = s_model(X, training = True)
		supervised_loss = sup_loss(y, sup_preds)

	supervised_grads = s_tape.gradient(supervised_loss, s_model.trainable_variables)
	s_opt.apply_gradients(zip(supervised_grads, s_model.trainable_variables))

	class_loss(supervised_loss)
	class_acc(y, sup_preds)

def train_unsup_step(X_real, X_gen, d_model, d_opt, disc_loss_metric, g_model, g_opt, gen_loss_metric):
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		gen_images = g_model(X_gen, training = True)

		real_output = d_model(X_real, training = True)
		fake_output = d_model(gen_images, training = True)

		gen_loss = generator_loss(fake_output)
		disc_loss = unsup_loss(real_output, fake_output)

	gen_grads = gen_tape.gradient(gen_loss, g_model.trainable_variables)
	disc_grads = disc_tape.gradient(disc_loss, d_model.trainable_variables)

	g_opt.apply_gradients(zip(gen_grads, g_model.trainable_variables))
	d_opt.apply_gradients(zip(disc_grads, d_model.trainable_variables))

	disc_loss_metric(disc_loss)
	gen_loss_metric(gen_loss)

# def train_gen_step(X, y, d_model, g_model, g_opt, gen_loss):

# 	with tf.GradientTape() as g_tape:
# 		gen_images = g_model(X, training = True)
# 		disc_preds = d_model(gen_images, training = False)
# 		generator_loss = gen_loss(y, disc_preds)
	
# 	generator_grads = g_tape.gradient(generator_loss, g_model.trainable_variables)
# 	g_opt.apply_gradients(zip(generator_grads, g_model.trainable_variables))

# 	gen_loss(generator_loss)


batch_size = 100
gen_dim = 100
nb_epochs = 20

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], -1).astype('float32')
X_train = (X_train - 127.5) / 127.5


batches_per_epoch = X_train.shape[0] // batch_size
nb_steps = batches_per_epoch * nb_epochs

half_batch_size = batch_size // 2
s_model, d_model = make_discriminator_models()
s_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)
d_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)

g_model = make_generator_model(gen_dim)
g_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)

class_loss_metric = tf.keras.metrics.Mean(name = 'classification_loss')
class_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name = 'classification_accuracy')

disc_loss_metric = tf.keras.metrics.Mean(name = 'discriminator_loss')
gen_loss_metric = tf.keras.metrics.Mean(name = 'generator_loss')

for i in range(1, nb_steps + 1):

	class_loss_metric.reset_states()
	class_acc_metric.reset_states()
	disc_loss_metric.reset_states()
	gen_loss_metric.reset_states()

	#Train s_model (softmax)
	X_sup, y_sup = select_supervised_samples(X_train, y_train)
	[X_sup_batch, y_sup_classes], _ = generate_real_samples(X_sup, y_sup, half_batch_size)
	train_sup_step(X_sup_batch, y_sup_classes, s_model, s_opt, class_loss_metric, class_acc_metric)

	#Train d_model/g_model (discriminator/generator)
	[X_unsup_real, _], y_unsup_real = generate_real_samples(X_train, y_train, half_batch_size)

	X_gen = generate_input_vector(half_batch_size, gen_dim)

	# X_gen, y_gen = generate_input_vector(batch_size, gen_dim), tf.ones(batch_size, 1)
	train_unsup_step(X_unsup_real, X_gen, d_model, d_opt, disc_loss_metric, g_model, g_opt, gen_loss_metric)

	template = 'Step {}, Classification Loss: {}, Classification Accuracy: {}, Discriminator Loss: {}, Generator Loss: {}'
	print(template.format(i,
			class_loss_metric.result(),
			class_acc_metric.result() * 100,
			disc_loss_metric.result(),
			gen_loss_metric.result()))



