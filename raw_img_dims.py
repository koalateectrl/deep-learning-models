import os
import nibabel as nib
import numpy as np
from collections import defaultdict

def get_path_list(noisy_path, clean_path):
    prefix_list = prefix_list = [f.split("noise")[0] for f in os.listdir(noisy_path) if not f.startswith('.')]
    prefix_list = np.sort(prefix_list)
    path_list = [(os.path.join(noisy_path, prefix + "noise.nii.gz"), os.path.join(clean_path, prefix + "clean.nii.gz")) for prefix in prefix_list]
    return path_list

def get_shape(one_tuple):
    noisy_img_np = np.array(nib.load(one_tuple[0]).dataobj)
    clean_img_np = np.array(nib.load(one_tuple[1]).dataobj)

    noisy_shape = noisy_img_np.shape
    clean_shape = clean_img_np.shape

    del noisy_img_np, clean_img_np

    return noisy_shape, clean_shape


noisy_path = '/data2/liztong/AI_rsFMRI/noise_rsFMRI'
clean_path = '/data2/liztong/AI_rsFMRI/clean_rsFMRI'

path_list = get_path_list(noisy_path, clean_path)

print(path_list)
shape_dict = defaultdict(int)

for idx, path_tuple in enumerate(path_list):
    if path_tuple[0] not in ['/data2/liztong/AI_rsFMRI/noise_rsFMRI/119732_LR_noise.nii.gz', '/data2/liztong/AI_rsFMRI/noise_rsFMRI/127630_LR_noise.nii.gz', '/data2/liztong/AI_rsFMRI/noise_rsFMRI/150423_LR_noise.nii.gz', '/data2/liztong/AI_rsFMRI/noise_rsFMRI/159946_LR_noise.nii.gz', '/data2/liztong/AI_rsFMRI/noise_rsFMRI/183337_LR_noise.nii.gz']:
    	noisy_shape, clean_shape = get_shape(path_tuple)
    	shape_dict[noisy_shape] += 1
    	print("Index: " + str(idx))
    	print(noisy_shape)
    	print(clean_shape)
    	print(shape_dict)

print(path_list[90], path_list[122], path_list[232], path_list[283], path_list[302])
