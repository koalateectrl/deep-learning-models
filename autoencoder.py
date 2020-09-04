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

STEPS_PER_EPOCH = 60000 // batch_size



def ConvolutionBlock(x, name, fms, params):
    x = tf.keras.layers.Conv2D(filters = fms, **params, name = name + "_conv0")(x)
    x = tfa.layers.InstanceNormalization(axis = -1, center = True, scale = True,
        beta_initializer = "random_uniform",
        gamma_initializer = "random_uniform", name = name + "_bn0")(x)
    x = tf.keras.layers.LeakyReLU(alpha = 0.1, name = name + "_relu0")(x)
    x = tf.keras.layers.Conv2D(filters = fms, **params, name = name + "_conv1")(x)
    x = tfa.layers.InstanceNormalization(axis = -1, center = True, scale = True,
        beta_initializer = "random_uniform",
        gamma_initializer = "random_uniform", name = name + "_bn1")(x)
    x = tf.keras.layers.LeakyReLU(alpha = 0.1, name = name)(x)
    return x


def unet_2D():

    learning_rate = 0.001
    use_upsampling = False

    input_shape = [16, 16, 1]
    data_format = "channels_last"
    concat_axis = -1

    inputs = K.layers.Input(shape = input_shape, name = "fMRImages")

    params = dict(kernel_size = (3, 3), activation = None,
        padding = "same", data_format = data_format,
        kernel_initializer = "he_uniform")

    fms = 32
    dropout = 0.3
    n_cl_out = 1

    params_trans = dict(data_format = data_format,
        kernel_size = (2, 2), strides = (2, 2),
        padding = "same")

    # BEGIN - Encoding path
    encodeA = ConvolutionBlock(inputs, "encodeA", fms, params)
    poolA = K.layers.MaxPooling2D(name = "poolA", pool_size=(2, 2))(encodeA)

    encodeB = ConvolutionBlock(poolA, "encodeB", fms * 2, params)
    poolB = K.layers.MaxPooling2D(name = "poolB", pool_size=(2, 2))(encodeB)

    encodeC = ConvolutionBlock(poolB, "encodeC", fms * 4, params)
    poolC = K.layers.MaxPooling2D(name = "poolC", pool_size=(2, 2))(encodeC)

    encodeD = ConvolutionBlock(poolC, "encodeD", fms * 8, params)
    poolD = K.layers.MaxPooling2D(name = "poolD", pool_size=(2, 2))(encodeD)

    encodeE = ConvolutionBlock(poolD, "encodeE", fms * 16, params)
    # END - Encoding path

    # BEGIN - Decoding path
    if use_upsampling:
        up = K.layers.UpSampling2D(name = "upE", size = (2, 2),
                                   interpolation="bilinear")(encodeE)
    else:
        up = K.layers.Conv2DTranspose(name = "transconvE", filters = fms * 8,
                                      **params_trans)(encodeE)
    concatD = K.layers.concatenate(
        [up, encodeD], axis = concat_axis, name = "concatD")

    decodeC = ConvolutionBlock(concatD, "decodeC", fms * 8, params)

    if use_upsampling:
        up = K.layers.UpSampling2D(name = "upC", size = (2, 2),
                                   interpolation = "bilinear")(decodeC)
    else:
        up = K.layers.Conv2DTranspose(name = "transconvC", filters = fms * 4,
                                      **params_trans)(decodeC)
    concatC = K.layers.concatenate(
        [up, encodeC], axis = concat_axis, name = "concatC")

    decodeB = ConvolutionBlock(concatC, "decodeB", fms * 4, params)

    if use_upsampling:
        up = K.layers.UpSampling2D(name = "upB", size = (2, 2),
                                   interpolation = "bilinear")(decodeB)
    else:
        up = K.layers.Conv2DTranspose(name = "transconvB", filters = fms * 2,
                                      **params_trans)(decodeB)
    concatB = K.layers.concatenate(
        [up, encodeB], axis = concat_axis, name = "concatB")

    decodeA = ConvolutionBlock(concatB, "decodeA", fms * 2, params)

    if use_upsampling:
        up = K.layers.UpSampling2D(name = "upA", size = (2, 2, 2),
                                   interpolation = "bilinear")(decodeA)
    else:
        up = K.layers.Conv2DTranspose(name = "transconvA", filters = fms,
                                      **params_trans)(decodeA)
    concatA = K.layers.concatenate(
        [up, encodeA], axis = concat_axis, name = "concatA")

    # END - Decoding path

    convOut = ConvolutionBlock(concatA, "convOut", fms, params)

    prediction = K.layers.Conv2D(name = "PredictionMask",
        filters = n_cl_out, kernel_size = (1, 1), 
        data_format = data_format,
        activation = "linear")(convOut)

    model = K.models.Model(inputs = [inputs], outputs = [prediction])

    model.summary()
    return model


def loss_fn(y_true, y_pred):
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)


# @tf.function
def train_step(images, labels, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_fn(images, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


# @tf.function
def test_step(test_ds, model, num_steps):
    num_eval_steps = num_steps // batch_size
    losses = 0.0
    classification_loss = 0.0
    for (batch, (x, y_seg)) in enumerate(test_ds.take(num_eval_steps)):
        logits = model(x, training = False)
        mae = loss_fn(y_seg, logits)
        classification_loss += tf.reduce_mean(mae)
        losses = classification_loss
        loss_tot = losses / batch

    return loss_tot, classification_loss

model = unet_2D()
start_epoch = 0


initial_learning_rate = 0.001
num_epochs = 200
steps_per_epoch = STEPS_PER_EPOCH
num_steps = int(num_epochs * steps_per_epoch)
print(num_steps)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
    decay_steps = 200 * steps_per_epoch,
    decay_rate = 0.1,
    staircase = True)
optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer)

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
test_loss = tf.keras.metrics.Mean(name = 'test_loss')


train_loss_results = []
train_accuracy_results = []
train_losses_np = []
test_losses_np = []

for (batch, (x, y)) in enumerate(train_ds.take(num_steps)):
#     #Optimize the model
    epoch = int(batch // steps_per_epoch)
    class_loss = train_step(x, y, model, optimizer)
    loss_value = class_loss
    train_loss(loss_value) # Add current batch loss
    train_losses_np.append(train_loss.result().numpy())

    if ((batch % 10) == 0):
        print("epoch: ", str(epoch), " loss: ", str(train_loss.result()))

    if (batch % steps_per_epoch == 0):
        test_losses, classification_loss = test_step(test_ds, model, num_steps)
        test_losses_np.append(test_losses)
        row = "epoch: " + str(epoch) + " train loss:" + str(train_loss.result().numpy()) + "auc1: " + str(test_losses)
        print(row)



#         checkpoint_path = "checkpoints_new/" + "unet2d_" + str(epoch+start_epoch) + ".h5"

        # row_ = checkpoint_path + " " + row
        # with open("testsetresults_2d" + ".csv", "a") as fd:
        #     fd.write(row_)
        #     fd.write("\n")




















