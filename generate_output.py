import tensorflow as tf
import time
from tensorflow import keras as K
#import tensorflow_addons as tfa
import os
import numpy as np
import nibabel as nib
#import cv2

from absl import flags
from absl import app
from absl import logging

flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_string('noisy_path', '/home/data2/liztong/AI_rsFMRI/Resampled/Resampled_noise', 'Directory with Noisy fMRI Images')
flags.DEFINE_string('clean_path', '/home/data2/liztong/AI_rsFMRI/Resampled/Resampled_clean_v2', 'Directory with Clean fMRI Images')
flags.DEFINE_string('ckpt_path', '/home/data2/samwwong/fMRI_denoising/checkpoints', 'Model Directory')
flags.DEFINE_string('output_path', '/home/data2/samwwong/fMRI_denoising/output', 'Test Outputs')

flags.DEFINE_integer('test_image', 300, 'Index of test image')

FLAGS = flags.FLAGS


def get_path_list():
    prefix_list = [f.split("noise")[0] for f in os.listdir(FLAGS.noisy_path) if not f.startswith('.')]
    prefix_list = np.sort(prefix_list)

    path_list = [(os.path.join(FLAGS.noisy_path, prefix + "noise_sub2_tp300.nii.gz"), 
                  os.path.join(FLAGS.clean_path, prefix + "clean_sub2_tp300_v2.nii.gz")) for prefix in prefix_list]
    return path_list

# def normalize(img_np):
#     '''Linearly normalizes voxel values between 0 and 1'''
#     min_val = np.min(img_np)
#     max_val = np.max(img_np)

#     return (img_np - min_val) / (max_val - min_val), min_val, max_val

def normalize(img_np):
    return img_np / 40000


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


def create_data_set(noisy_img_path):
    img = nib.load(noisy_img_path)
    affine = img.affine
    header = img.header
    img_np = np.array(img.dataobj)
    img_np = preprocess(img_np)

    generate_noisy(img_np, affine, header)

    noisy_img = normalize(img_np)

    noisy_img_expanded_dims = np.expand_dims(noisy_img, axis = 4)

    noisy_len = noisy_img_expanded_dims.shape[0]
    del noisy_img

    noisy_ds = tf.data.Dataset.from_tensor_slices(noisy_img_expanded_dims).batch(FLAGS.batch_size)
    del noisy_img_expanded_dims
    return noisy_ds, noisy_len, affine, header


def unnormalize(img_np):
    return img_np * 40000

# def unnormalize(img_np, min_val, max_val):
    # return img_np * (max_val - min_val) + min_val

def generate_noisy(img_np, affine, header):
    img_np = unnormalize(img_np)
    img_np = np.moveaxis(img_np, 0, -1)
    print(img_np.shape)
    noisy_img = nib.Nifti1Image(img_np, affine, header)
    nib.save(noisy_img, os.path.join(FLAGS.output_path, "noisy_output.nii.gz"))


def generate_clean(clean_img_path):
    img = nib.load(clean_img_path)
    affine = img.affine
    header = img.header
    img_np = np.array(img.dataobj)
    img_np = unnormalize(img_np)
    img_np = preprocess(img_np)

    clean_img = np.moveaxis(img_np, 0, -1)
    print(clean_img.shape)
    clean_img = nib.Nifti1Image(clean_img, affine, header)
    nib.save(clean_img, os.path.join(FLAGS.output_path, "clean_output.nii.gz"))



def generate_output(images, model):
    predictions = model(images, training = False)
    return predictions


def main(unused_argv):
    path_list = get_path_list()
    print(path_list[FLAGS.test_image])

    noisy_ds, noisy_len, affine, header = create_data_set(path_list[FLAGS.test_image][0])

    for i in range(0, 11, 5):
        checkpoint_path = os.path.join(FLAGS.ckpt_path, "unet3d_" + str(i))
        model = tf.keras.models.load_model(checkpoint_path)

        output_img = np.zeros((noisy_len, 64, 64, 64, 1))

        for idx, images in enumerate(noisy_ds):
            print(idx)
            one_clean_img = generate_output(images, model)
            output_img[idx, :, :, :, :] = one_clean_img

        output_img = np.squeeze(output_img)
        output_img = unnormalize(output_img)
        output_img = np.moveaxis(output_img, 0, -1)
        print(output_img.shape)
        output_img = nib.Nifti1Image(output_img, affine, header)

        nib.save(output_img, os.path.join(FLAGS.output_path, "model_output_" + str(i) + ".nii.gz"))

        del output_img

    generate_clean(path_list[FLAGS.test_image][1])


if __name__ == '__main__':
    app.run(main)



