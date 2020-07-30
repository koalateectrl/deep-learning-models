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

def make_generator_model(latent_dim = 100):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(128 * 7 * 7, use_bias = False,  input_shape = (latent_dim, )))
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))

	model.add(tf.keras.layers.Reshape((7, 7, 128)))
	assert model.output_shape == (None, 7, 7, 128)

	model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides = (2,2), padding = 'same', use_bias = False))
	assert model.output_shape == (None, 14, 14, 128)
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))

	model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides = (2,2), padding = 'same', use_bias = False))
	assert model.output_shape == (None, 28, 28, 128)
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))

	model.add(tf.keras.layers.Conv2D(1, (7, 7), padding = 'same', use_bias = False, activation = 'tanh'))
	assert model.output_shape == (None, 28, 28, 1)

	return model

def make_base_discriminator_model(input_shape = (28, 28, 1), n_classes = 10):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same', input_shape = input_shape))
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))

	model.add(tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same'))
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))

	model.add(tf.keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same'))
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.2))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dropout(0.4))

	model.add(tf.keras.layers.Dense(n_classes))

	return model

def d_unsup_loss(real_output, fake_output):
	sigmoid_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)

	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_unsup_loss = real_loss + fake_loss
	return total_unsup_loss

def d_sup_loss(true_classes, pred_classes):
	cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
	return (cce_loss(true_classes, pred_classes))

def d_loss(real_output, fake_output, true_classes, pred_classes):
	return d_unsup_loss(real_output, fake_output) + d_sup_loss(true_classes, pred_classes)


def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_sup_step(X, y, model, opt):
	with tf.GradientTape() as tape:
		sup_preds = model(X, training = True)
		sup_loss = d_sup_loss(y, sup_preds)

	sup_grads = tape.gradient(sup_loss, model.trainable_variables)
	opt.apply_gradients(zip(sup_grads, model.trainable_variables))

	return

batch_size = 100

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], -1).astype('float32')
X_train = (X_train - 127.5) / 127.5


half_batch_size = batch_size // 2
d_model = make_base_discriminator_model()
d_sup_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)


X_sup, y_sup = select_supervised_samples(X_train, y_train)
[X_sup_batch, y_sup_classes], _ = generate_real_samples(X_sup, y_sup, half_batch_size)
train_sup_step(X_sup_batch, y_sup_classes, d_model, d_sup_opt)



# [X_sup_batch, y], y_is_real = generate_real_samples(X_train, y_train, half_batch_size)





# BATCH_SIZE = 256
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train)).shuffle(60000).batch(BATCH_SIZE)

# d_unsup_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)

# g_opt = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)





