import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def create_patch(image, x, y, patch_size):
	patch_image = np.array(image, copy = True)
	patch_image[x: x + patch_size, y: y + patch_size, :] = 127.5
	return patch_image


def test_step(images, model):
	prediction = model(images, training = False)
	return prediction

#Loading in the image
path_to_image = "/Users/samwwong/Downloads/dog.jpg"
image = plt.imread(path_to_image)
h, w, c = image.shape

#Preprocessing the image
IMG_SIZE = 224
image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
image = tf.keras.applications.vgg16.preprocess_input(image)
image = tf.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

#Modeling and prediction
vgg_model = tf.keras.applications.VGG16(include_top = True, weights = 'imagenet')
prediction = test_step(image, vgg_model)

#Results
true_max_idx = tf.math.argmax(prediction, axis = 1)[0]
true_max_val = prediction[0, true_max_idx]

print(true_max_val)

patch_size = 32
occlusion_map = np.zeros((image.shape[1], image.shape[2]))

#Run Occlusion by obscuring part of the image at a time
for x_val in range(0, image.shape[1], patch_size):
	for y_val in range(0, image.shape[2], patch_size):
		print(x_val, y_val)
		patch_image = create_patch(image, x_val, y_val, patch_size)
		occl_preds = test_step(patch_image, model)
		occl_cls_prob = occl_preds[0, true_max_idx]
		print(occl_cls_prob)

		occlusion_map[x_val:x_val + patch_size, y_val: y_val + patch_size] = occl_cls_prob


print(type(occlusion_map))
print(occlusion_map.shape)
print(occlusion_map)

plt.imshow(occlusion_map)
plt.show()

















