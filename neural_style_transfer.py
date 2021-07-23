import tensorflow as tf
import numpy as np
from PIL import Image

from matplotlib import image
from matplotlib import pyplot as plt

style_path = "../Downloads/chinese.jpg"
style_path_2 = "../Downloads/chinese2.png"
content_path = "../Downloads/paloalto.jpeg"
content_path_2 = "../Downloads/turtle.jpg"

content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
				'block2_conv1',
				'block3_conv1',
				'block4_conv1',
				'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def view_image(path):
	data = image.imread(path)
	print(data.dtype)
	print(data.shape)
	plt.imshow(data)
	plt.show()

def load_image(path):
	data = image.imread(path)
	return data

def preprocess(data):
	vgg_process_data = tf.keras.applications.vgg19.preprocess_input(data)
	resized_data = tf.image.resize(vgg_process_data, (224, 224))
	return resized_data


content_image = load_image(content_path)
content_image = preprocess(content_image)
content_image = tf.reshape(content_image, (-1, content_image.shape[0], content_image.shape[1], content_image.shape[2]))

style_image = load_image(style_path)
style_image = preprocess(style_image)
style_image = tf.reshape(style_image, (-1, style_image.shape[0], style_image.shape[1], style_image.shape[2]))

def vgg_layers(layer_names):
	vgg = tf.keras.applications.VGG19(include_top = False, weights = 'imagenet')
	vgg.trainable = False

	outputs = [vgg.get_layer(name).output for name in layer_names]
	model = tf.keras.Model(inputs = [vgg.inputs], outputs = outputs)
	return model

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image)

for name, output in zip(style_layers, style_outputs):
	print(name)
	print(f"shape: {output.numpy().shape}")
	print(f"min: {output.numpy().min()}")
	print(f"max: {output.numpy().max()}")
	print(f"mean: {output.numpy().mean()}")
	print()

def gram_matrix(input_tensor):
	result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
	input_shape = tf.shape(input_tensor)
	num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
	return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
	def __init__(self, style_layers, content_layers):
		super(StyleContentModel, self).__init__()
		self.vgg = vgg_layers(style_layers)
		self.style_layers = style_layers
		self.content_layers = content_layers
		self.num_style_layers = len(style_layers)
		self.vgg.trainable = False

	def call(self, inputs):
		inputs = inputs*255.0
		preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
		outputs = self.vgg(preprocessed_input)
		style_outputs, content_outputs = (outputs[:self.num_style_layers],
			outputs[self.num_style_layers:])

		style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

		content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}

		style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}

		return {'content':content_dict, 'style':style_dict}

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

print('Styles:')
for name, output in sorted(results['style'].items()):
	print("  ", name)
	print("    shape: ", output.numpy().shape)
	print("    min: ", output.numpy().min())
	print("    max: ", output.numpy().max())
	print("    mean: ", output.numpy().mean())
	print()

print("Contents:")
for name, output in sorted(results['content'].items()):
	print("  ", name)
	print("    shape: ", output.numpy().shape)
	print("    min: ", output.numpy().min())
	print("    max: ", output.numpy().max())
	print("    mean: ", output.numpy().mean())




# model = create_model()
# style_features, content_features = get_feature_representations(model, content_path, style_path)
# gram_style_features = [create_gram_matrix(style_features) for style in style_features]

# init_image = preprocess(load_image(content_path))
# optimizer = tf.keras.optimizers.Adam(learning_rate = 5, beta1 = 0.99)

# iter_count = 1

# best_loss, best_img = float('inf'), None

# loss_weights = (style_weight, content_weight)




