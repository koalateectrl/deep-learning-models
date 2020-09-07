import tensorflow as tf
import time
from tensorflow import keras as K
import tensorflow_addons as tfa

from absl import flags
from absl import app
from absl import logging

batch_size = 256

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], -1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype("float32")
X_train = tf.image.resize(X_train, [16, 16])
X_test = tf.image.resize(X_test, [16, 16])

print(X_train.shape)
print(X_test.shape)

TRAIN_LENGTH = X_train.shape[0]
TEST_LENGTH = X_test.shape[0]

BUFFER_SIZE = batch_size * 10

train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train)).shuffle(BUFFER_SIZE, seed = 0).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, X_test)).batch(batch_size)

def loss_fn(y_true, y_pred):
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)

def test_step(images, labels, model, test_loss):
    with tf.GradientTape() as tape:
        predictions = model(images, training = False)
        loss = loss_fn(labels, predictions)

    test_loss(loss)

checkpoint_path = "checkpoints/unet2d_" + str(10)
model = tf.keras.models.load_model(checkpoint_path)

test_loss = tf.keras.metrics.Mean(name = 'test_loss')



for test_images, test_labels in test_ds:
    test_step(test_images, test_labels, model, test_loss)

template = 'Test Loss {}'
print(template.format(test_loss.result()))



