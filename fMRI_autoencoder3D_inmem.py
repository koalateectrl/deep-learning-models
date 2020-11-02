import tensorflow as tf
import time
from tensorflow import keras as K
import tensorflow_addons as tfa
import os
import numpy as np
import nibabel as nib
import cv2

from absl import flags
from absl import app
from absl import logging

flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for training')
flags.DEFINE_string('ckpt_path', 'checkpoints/unet3d_', 'Checkpoint Directory')
flags.DEFINE_integer('seed', 0, 'Seed for shuffling training batch')

flags.DEFINE_string('noisy_path', 'home/data2/liztong/AI_rsFMRI/Resampled/Resampled_noise', 'Directory with Noisy fMRI Images')
flags.DEFINE_string('clean_path', '/home/data2/liztong/AI_rsFMRI/Resampled/Resampled_clean_v2', 'Directory with Clean fMRI Images')
flags.DEFINE_integer('train_len', 400, 'Number of 3D-images per training example')



FLAGS = flags.FLAGS


def get_img_paths():
    prefix_list = [f.split("noise")[0] for f in os.listdir(noisy_path) if not f.startswith('.')]
    prefix_list = np.sort(prefix_list)

    path_list = [(os.path.join(noisy_path, prefix + "noise_sub2_tp300.nii.gz"), 
                  os.path.join(clean_path, prefix + "clean_sub2_tp300_v2.nii.gz")) for prefix in prefix_list]
    return path_list

def normalize(img_np):
    return img_np / 40000


# def normalize(img_np):
#     '''Linearly normalizes voxel values between 0 and 1'''
#     min_val = np.min(img_np)
#     max_val = np.max(img_np)

#     return (img_np - min_val) / (max_val - min_val)


# def resize(img_np, height = 128, width = 128):
#     img_np = np.moveaxis(img_np, -1, 0)
#     depth = img_np.shape[3]

#     resized_np = np.zeros((len(img_np), width, height, depth))
#     for idx in range(len(img_np)):
#         img = img_np[idx, :, :, :]
#         img_res = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
#         resized_np[idx, :, :, :] = img_res

#     # to add the 92nd depth layer
#     temp = np.zeros((resized_np.shape[0], resized_np.shape[1], resized_np.shape[2], 5))
#     resized_np = np.concatenate((temp, resized_np), axis = 3)
#     return resized_np

def preprocess(img_np):
    img_np = np.pad(img_np, ((9, 9), (5, 4), (9, 9), (0, 0)), 'constant')
    img_np = np.moveaxis(img_np, -1, 0)
    return normalize(img_np)


def create_data_set(one_tuple):
    print(one_tuple[0])
    # Create X_train
    img_np_noise = np.array(nib.load(one_tuple[0]).dataobj)
    X_train = preprocess(img_np_noise)

    train_len = X_train.shape[0] 
    X_expanded_dims = np.expand_dims(X_train, axis = 4)
    del X_train, img_np_noise

    # Create Y_train
    img_np_clean = np.array(nib.load(one_tuple[1]).dataobj)

    Y_train = preprocess(img_np_clean)

    Y_expanded_dims = np.expand_dims(Y_train, axis = 4)
    del Y_train, img_np_clean

    print(X_expanded_dims.shape, Y_expanded_dims.shape)

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

    input_shape = [64, 64, 64, 1]
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

    path_list = get_path_list()
    
    model = unet_3D()

    steps_per_epoch = FLAGS.train_len // FLAGS.batch_size

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(FLAGS.learning_rate,
        decay_steps = steps_per_epoch,
        decay_rate = 0.1,
        staircase = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer)

    train_loss = tf.keras.metrics.Mean(name = 'train_loss')
    test_loss = tf.keras.metrics.Mean(name = 'test_loss')

    for epoch in range(1, FLAGS.num_epochs + 1):
        for idx, one_tuple in enumerate(path_list[:2]):
            if one_tuple[0] not in ['/home/data2/liztong/AI_rsFMRI/noise_rsFMRI/119732_LR_noise.nii.gz', 
            '/home/data2/liztong/AI_rsFMRI/noise_rsFMRI/127630_LR_noise.nii.gz', 
            '/home/data2/liztong/AI_rsFMRI/noise_rsFMRI/150423_LR_noise.nii.gz', 
            '/home/data2/liztong/AI_rsFMRI/noise_rsFMRI/159946_LR_noise.nii.gz', 
            '/home/data2/liztong/AI_rsFMRI/noise_rsFMRI/183337_LR_noise.nii.gz']:

                train_ds, train_len = create_data_set(one_tuple)

                start_time = time.time()

                train_loss.reset_states()
                test_loss.reset_states()

                for (step, (images, labels)) in enumerate(train_ds):
                    print("STEP: " + str(step))
                    train_step(images, labels, model, optimizer, train_loss)

                end_time = time.time()
                logging.info(f"Epoch {epoch} time in seconds: {end_time - start_time}")

                del train_ds

                if idx % 1 == 0:
                    test_ds, test_len = create_data_set(path_list[150])

                    for test_images, test_labels in test_ds:
                        test_step(test_images, test_labels, model, test_loss)

                    del test_ds

                    # template = 'Epoch {}, Loss: {}, Test Loss {}'
                    # print(template.format(epoch,
                    #     train_loss.result(),
                    #     test_loss.result()))

                    checkpoint_path = FLAGS.ckpt_path + str(idx)
                    model.save(checkpoint_path)

                    row = "epoch: " + str(idx) + " train loss:" + str(train_loss.result()) + "test loss:" + str(test_loss.result())
                    print(row)
                    with open('modelresults_3d_new.csv', 'a') as fd:
                        fd.write(row)
                        fd.write("\n")


if __name__ == "__main__":
    app.run(main)




















