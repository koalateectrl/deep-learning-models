import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def load_images(path, n = 0):
	X = []
	y = []
	cat_dict = {}
	lang_dict = {}
	curr_y = n

	for alphabet in listdir_nohidden(path):
		print("loading alphabet: " + alphabet)
		lang_dict[alphabet] = [curr_y, None]
		alphabet_path = os.path.join(path, alphabet)

		for letter in listdir_nohidden(alphabet_path):
			cat_dict[curr_y] = (alphabet, letter)
			category_images = []
			letter_path = os.path.join(alphabet_path, letter)

			for filename in listdir_nohidden(letter_path):
				image_path = os.path.join(letter_path, filename)
				image = plt.imread(image_path)
				category_images.append(image)
				y.append(curr_y)
			try:
				X.append(np.stack(category_images))
			except ValueError as e:
				print(e)
				print("error - category_images:", category_images)
			curr_y += 1
			lang_dict[alphabet][1] = curr_y - 1
	y = np.vstack(y)
	X = np.stack(X)
	return X, y, lang_dict



train_folder_path = '/Users/samwwong/Downloads/images_background'
X_train, y_train, class_dict = load_images(train_folder_path)





def get_batch(data_set, batch_size):
	n_classes, n_examples, h, w = data_set.shape

	categories = np.random.choice(n_classes, size = (batch_size, ), replace = False)

	pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

	targets = np.zeros((batch_size,))

	targets[batch_size//2:] = 1

	for i in range(batch_size):
		category = categories[i]
		idx_1 = np.random.randint(0, n_examples)
		pairs[0][i, :, :, :] = data_set[category, idx_1].reshape(h, w, 1)
		
		idx_2 = np.random.randint(0, n_examples)
		if i >= batch_size // 2:
			category_2 = category
		else:
			category_2 = (category + np.random.randint(1, n_classes)) % n_classes

		pairs[1][i, :, :, :] = data_set[category_2, idx_2].reshape(h, w, 1)

	return pairs, targets

pairs, targets = get_batch(X_train, 10)

print(pairs)
print(targets)
print(pairs.shape)















