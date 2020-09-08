import numpy as np
import os
import nibabel as nib
import cv2

def get_img_paths(noisy_path, clean_path):
    prefix_list = [f.split("noise")[0] for f in os.listdir(noisy_path) if not f.startswith('.')]
    prefix_list = np.sort(prefix_list)

    path_list = [(os.path.join(noisy_path, prefix + "noise.nii.gz"), os.path.join(clean_path, prefix + "clean.nii.gz")) for prefix in prefix_list]
    return path_list

def nifti_to_np(path):
    img_np = np.array(nib.load(path).dataobj)
    img_np = resize(img_np)
    return normalize(img_np)

def normalize(img_np):
    '''Linearly normalizes voxel values between 0 and 1'''
    min_val = np.min(img_np)
    max_val = np.max(img_np)

    return (img_np - min_val) / (max_val - min_val)

def resize(img_np, height = 128, width = 128):
    img_np = np.moveaxis(img_np, -1, 0)
    depth = img_np.shape[3]

    resized_np = np.zeros((len(img_np), width, height, depth))
    for idx in range(len(img_np)):
        img = img_np[idx, :, :, :]
        img_res = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        resized_np[idx, :, :, :] = img_res
    return resized_np


def save_to_nifti(img_np, path):
    nib.save(img_np, path)

