import utils
import numpy as np

idx = 0

noisy_path = '/home/data2/liztong/AI_rsFMRI/noise_rsFMRI'
clean_path = '/home/data2/liztong/AI_rsFMRI/clean_rsFMRI'

path_list = utils.get_img_paths(noisy_path, clean_path)

one_tuple = path_list[idx]

noisy_img_np = utils.nifti_to_np(one_tuple[0])
clean_img_np = utils.nifti_to_np(one_tuple[1])

print(noisy_img_np.shape)
print(clean_img_np.shape)

np.save("preprocessed_images/noisy_" + str(idx) + "_1200x128x128x96.npy", noisy_img_np)
np.save("preprocessed_images/clean_" + str(idx) + "_1200x128x128x96.npy", clean_img_np)
