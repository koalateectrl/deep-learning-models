import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

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


# with open("X_train.pickle", "wb") as handle:
# 	pickle.dump(X_train, handle)



def get_batch(data_set, batch_size):
    num_classes, num_examples, height, width = data_set.shape
    categories = np.random.choice(num_classes, size = batch_size, replace = False)
    
    #Create pairs(2 images put together) and targets (half are the same character and half are different)
    pairs = np.zeros((2, batch_size, height, width))
    target = np.zeros((batch_size))
    target[batch_size // 2:] = 1
    
    for i in range(batch_size):
        category = categories[i]
        idx_1 = np.random.randint(0, num_examples)
        pairs[0][i][:][:] = data_set[category][idx_1]
        
        idx_2 = np.random.randint(0, num_examples)
        if i >= batch_size // 2:
            category_2 = category
        else:
            category_2 = (category + np.random.randint(1, num_classes)) % num_classes
        
        pairs[1][i][:][:] = data_set[category_2][idx_2]
    
    return pairs, target

pairs, targets = get_batch(X_train, 10)
















