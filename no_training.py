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
flags.DEFINE_string('noisy_path', '/home/data2/liztong/AI_rsFMRI/Resampled/Resampled_noise', 'Directory with Noisy fMRI Images')
flags.DEFINE_string('clean_path', '/home/data2/liztong/AI_rsFMRI/Resampled/Resampled_clean_v2', 'Directory with Clean fMRI Images')
flags.DEFINE_string('output_path', '/home/data2/samwwong/fMRI_denoising/csf_removed/output', 'Test Outputs')

flags.DEFINE_integer('train_len', 400, 'Number of 3D-images per training example')
flags.DEFINE_integer('test_image', 42, 'Index of test image')

FLAGS = flags.FLAGS


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


def get_path_list():
    prefix_list = [f.split("noise")[0] for f in os.listdir(FLAGS.noisy_path) if not f.startswith('.')]
    prefix_list = np.sort(prefix_list)

    path_list = [(os.path.join(FLAGS.noisy_path, prefix + "noise_sub2_tp300.nii.gz"), 
                  os.path.join(FLAGS.clean_path, prefix + "clean_sub2_tp300_v2.nii.gz")) for prefix in prefix_list]
    return path_list

def resize(img_np):
    img_np = np.pad(img_np, ((9, 9), (5, 4), (9, 9), (0, 0)), 'constant')
    img_np = np.moveaxis(img_np, -1, 0)
    return img_np


def normalize(img_np):
    return img_np / 40000

def unnormalize(img_np):
    return img_np * 40000


def create_data_set(noisy_img_path):
    img = nib.load(noisy_img_path)
    affine = img.affine
    header = img.header
    img_np = np.array(img.dataobj)
    img_np = resize(img_np)

    noisy_img = normalize(img_np)#, min_val, max_val = normalize(img_np)

    noisy_img_expanded_dims = np.expand_dims(noisy_img, axis = 4)

    noisy_len = noisy_img_expanded_dims.shape[0]
    del noisy_img

    noisy_ds = tf.data.Dataset.from_tensor_slices(noisy_img_expanded_dims).batch(FLAGS.batch_size)
    del noisy_img_expanded_dims
    return noisy_ds, noisy_len, affine, header#, min_val, max_val


def generate_output(images, model):
    predictions = model(images, training = False)
    return predictions


def main(unused_argv):
    path_list = get_path_list()
    model = unet_3D()

    noisy_ds, noisy_len, affine, header = create_data_set(path_list[FLAGS.test_image][0]) #, min_val, max_val = create_data_set(path_list[FLAGS.test_image][0])

    output_img = np.zeros((noisy_len, 64, 64, 64, 1))

    for idx, images in enumerate(noisy_ds):
        print(idx)
        one_clean_img = generate_output(images, model)
        output_img[idx, :, :, :, :] = one_clean_img

    output_img = np.squeeze(output_img)
    output_img = unnormalize(output_img)#, min_val, max_val)
    output_img = np.moveaxis(output_img, 0, -1)
    print(output_img.shape)
    output_img = nib.Nifti1Image(output_img, affine, header)

    nib.save(output_img, os.path.join(FLAGS.output_path, "no_training.nii.gz"))


if __name__ == "__main__":
    app.run(main)


