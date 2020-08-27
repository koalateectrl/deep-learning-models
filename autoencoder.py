import tensorflow as tf
import time

from absl import flags
from absl import app
from absl import logging

batch_size = 256

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], -1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], -1).astype("float32")
train_ds = tf.data.Dataset.from_tensor_slices(X_train).shuffle(60000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices(X_test).shuffle(10000).batch(batch_size)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding = 'same')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size = (2, 2), padding = 'same')
        self.conv2 = tf.keras.layers.Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu', padding = 'same')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size = (2, 2), padding = 'same')
        self.conv3 = tf.keras.layers.Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu', padding = 'same')

        self.bottleneck = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), padding = 'same')

        self.conv4 = tf.keras.layers.Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu', padding = 'same')
        self.upsample1 = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.conv5 = tf.keras.layers.Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu', padding = 'same')
        self.upsample2 = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.conv6 = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding = 'valid')
        self.upsample3 = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.conv7 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding = 'same', activation = 'sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bottleneck(x)
        x = self.conv4(x)
        x = self.upsample1(x)
        x = self.conv5(x)
        x = self.upsample2(x)
        x = self.conv6(x)
        x = self.upsample3(x)
        x = self.conv7(x)
        return x

    def model(self, input_shape = (28, 28, 1)):
        x = tf.keras.Input(shape = input_shape)
        return tf.keras.Model(inputs = x, outputs = self.call(x))

test_model = MyModel()
print(test_model.model().summary())


def loss_fn(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)

@tf.function
def train_step(images, model, optimizer, train_loss):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_fn(images, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


@tf.function
def test_step(images, model, test_loss):
    predictions = model(images, training = False)
    loss = loss_fn(images, predictions)

    test_loss(loss)


learning_rate = 0.001
epochs = 50
model = MyModel()
optimizer = tf.keras.optimizers.SGD(lr = learning_rate)

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
test_loss = tf.keras.metrics.Mean(name = 'test_loss')

for epoch in range(1, epochs + 1):
    start_time = time.time()
    train_loss.reset_states()
    test_loss.reset_states()

    for images in train_ds:
        train_step(images, model, optimizer, train_loss)

    end_time = time.time()
    logging.info(f"Epoch {epoch} time in seconds: {end_time - start_time}")

    for test_images in test_ds:
        test_step(images, model, test_loss)

    template = 'Epoch {}, Loss: {}, Test Loss {}'
    print(template.format(epoch, 
        train_loss.result(), 
        test_loss.result()))




















