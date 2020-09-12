import tensorflow as tf
import time
from tensorflow import keras as K
import tensorflow_addons as tfa
import os
import numpy as np
import nibabel as nib

from absl import flags
from absl import app
from absl import logging

flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('num_epochs', 5, 'Number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for training')
flags.DEFINE_string('ckpt_path', 'checkpoints/unet2d_', 'Checkpoint Directory')
flags.DEFINE_integer('seed', 0, 'Seed for shuffling training batch')

# flags.DEFINE_string('noisy_path', '/home/data2/liztong/AI_rsFMRI/noise_rsFMRI', 'Directory with Noisy fMRI Images')
# flags.DEFINE_string('clean_path', '/home/data2/liztong/AI_rsFMRI/clean_rsFMRI', 'Directory with Clean fMRI Images')

flags.DEFINE_string('img_path', '/home/data2/samwwong/fMRI_denoising/preprocessing/preprocessed_images', 'Path to the preprocessed data')

FLAGS = flags.FLAGS



def create_data_set():

    X_train = np.load(FLAGS.img_path + "/noisy_0_1200x128x128x96.npy", allow_pickle=True)
    train_len = X_train.shape[0]
    X_expanded_dims = np.expand_dims(X_train, axis = 4)
    del X_train

    Y_train = np.load(FLAGS.img_path + "/clean_0_1200x128x128x96.npy", allow_pickle=True)
    Y_expanded_dims = np.expand_dims(Y_train, axis = 4)
    del Y_train

    print("DONE LOADING")

    BUFFER_SIZE = FLAGS.batch_size * 10

    train_ds = tf.data.Dataset.from_tensor_slices((X_expanded_dims, Y_expanded_dims)).shuffle(BUFFER_SIZE, seed = FLAGS.seed).batch(FLAGS.batch_size)

    del X_expanded_dims, Y_expanded_dims
    return train_ds, train_len



def ConvolutionBlock(x, name, fms, params):
    x = tf.keras.layers.Conv3D(filters = fms, **params, name = name + "_conv0")(x)
    x = tfa.layers.InstanceNormalization(axis = -1, center = True, scale = True,
        beta_initializer = "random_uniform",
        gamma_initializer = "random_uniform", name = name + "_bn0")(x)
    x = tf.keras.layers.LeakyReLU(alpha = 0.1, name = name + "_relu0")(x)
    x = tf.keras.layers.Conv3D(filters = fms, **params, name = name + "_conv1")(x)
    x = tfa.layers.InstanceNormalization(axis = -1, center = True, scale = True,
        beta_initializer = "random_uniform",
        gamma_initializer = "random_uniform", name = name + "_bn1")(x)
    x = tf.keras.layers.LeakyReLU(alpha = 0.1, name = name)(x)
    return x


def unet_3D():

    use_upsampling = False

    input_shape = [128, 128, 96, 1]
    data_format = "channels_last"
    concat_axis = -1

    inputs = K.layers.Input(shape = input_shape, name = "fMRImages")

    params = dict(kernel_size = (3, 3, 3), activation = None,
        padding = "same", data_format = data_format,
        kernel_initializer = "he_uniform")

    fms = 32
    dropout = 0.3
    n_cl_out = 1

    params_trans = dict(data_format = data_format,
        kernel_size = (2, 2, 2), strides = (2, 2, 2),
        padding = "same")

    # BEGIN - Encoding path
    encodeA = ConvolutionBlock(inputs, "encodeA", fms, params)
    poolA = K.layers.MaxPooling3D(name = "poolA", pool_size=(2, 2 ,2))(encodeA)

    encodeB = ConvolutionBlock(poolA, "encodeB", fms * 2, params)
    poolB = K.layers.MaxPooling3D(name = "poolB", pool_size=(2, 2, 2))(encodeB)

    encodeC = ConvolutionBlock(poolB, "encodeC", fms * 4, params)
    poolC = K.layers.MaxPooling3D(name = "poolC", pool_size=(2, 2, 2))(encodeC)

    encodeD = ConvolutionBlock(poolC, "encodeD", fms * 8, params)
    poolD = K.layers.MaxPooling3D(name = "poolD", pool_size=(2, 2, 2))(encodeD)

    encodeE = ConvolutionBlock(poolD, "encodeE", fms * 16, params)
    # END - Encoding path

    # BEGIN - Decoding path
    if use_upsampling:
        up = K.layers.UpSampling3D(name = "upE", size = (2, 2, 2),
                                   interpolation="bilinear")(encodeE)
    else:
        up = K.layers.Conv3DTranspose(name = "transconvE", filters = fms * 8,
                                      **params_trans)(encodeE)
    concatD = K.layers.concatenate(
        [up, encodeD], axis = concat_axis, name = "concatD")

    decodeC = ConvolutionBlock(concatD, "decodeC", fms * 8, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name = "upC", size = (2, 2, 2),
                                   interpolation = "bilinear")(decodeC)
    else:
        up = K.layers.Conv3DTranspose(name = "transconvC", filters = fms * 4,
                                      **params_trans)(decodeC)
    concatC = K.layers.concatenate(
        [up, encodeC], axis = concat_axis, name = "concatC")

    decodeB = ConvolutionBlock(concatC, "decodeB", fms * 4, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name = "upB", size = (2, 2, 2),
                                   interpolation = "bilinear")(decodeB)
    else:
        up = K.layers.Conv3DTranspose(name = "transconvB", filters = fms * 2,
                                      **params_trans)(decodeB)
    concatB = K.layers.concatenate(
        [up, encodeB], axis = concat_axis, name = "concatB")

    decodeA = ConvolutionBlock(concatB, "decodeA", fms * 2, params)

    if use_upsampling:
        up = K.layers.UpSampling3D(name = "upA", size = (2, 2, 2),
                                   interpolation = "bilinear")(decodeA)
    else:
        up = K.layers.Conv3DTranspose(name = "transconvA", filters = fms,
                                      **params_trans)(decodeA)
    concatA = K.layers.concatenate(
        [up, encodeA], axis = concat_axis, name = "concatA")

    # END - Decoding path

    convOut = ConvolutionBlock(concatA, "convOut", fms, params)

    prediction = K.layers.Conv3D(name = "PredictionMask",
        filters = n_cl_out, kernel_size = (1, 1, 1), 
        data_format = data_format,
        activation = "linear")(convOut)

    model = K.models.Model(inputs = [inputs], outputs = [prediction])

    model.summary()
    return model


def loss_fn(y_true, y_pred):
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)


# @tf.function
def train_step(images, labels, model, optimizer, train_loss):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


# @tf.function
def test_step(images, labels, model, test_loss):
    with tf.GradientTape() as tape:
        predictions = model(images, training = False)
        loss = loss_fn(labels, predictions)

    test_loss(loss)


def main(unused_argv):

    train_ds, train_len = create_data_set()
    test_ds, test_len = create_data_set()

    model = unet_3D()
    steps_per_epoch = train_len // FLAGS.batch_size

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(FLAGS.learning_rate,
        decay_steps = steps_per_epoch,
        decay_rate = 0.1,
        staircase = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer)

    train_loss = tf.keras.metrics.Mean(name = 'train_loss')
    test_loss = tf.keras.metrics.Mean(name = 'test_loss')


    for epoch in range(1, FLAGS.num_epochs + 1):
        start_time = time.time()

        train_loss.reset_states()
        test_loss.reset_states()

        for (step, (images, labels)) in enumerate(train_ds):
            print("STEP: " + str(step))
            train_step(images, labels, model, optimizer, train_loss)

        end_time = time.time()
        logging.info(f"Epoch {epoch} time in seconds: {end_time - start_time}")

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels, model, test_loss)

        template = 'Epoch {}, Loss: {}, Test Loss {}'
        print(template.format(epoch,
            train_loss.result(),
            test_loss.result()))

        checkpoint_path = FLAGS.ckpt_path + str(epoch)
        model.save(checkpoint_path)



if __name__ == "__main__":
    app.run(main)




















