import tensorflow as tf
import numpy as np

from collections import Counter


"""
Get Data (fashion_MNIST)

Since we want to create a simple image classification problem, let's just use the fashion MNIST dataset.

The dataset contains black and white images of fashion items and the goal is to classify what kind of item they are.


Label and Description:
- 0	T-shirt/top
- 1	Trouser
- 2	Pullover
- 3	Dress
- 4	Coat
- 5	Sandal
- 6	Shirt
- 7	Sneaker
- 8	Bag
- 9	Ankle boot
"""

class DataLoader():

	def __init__(self, num_shot, num_shot_eval, num_classes):
		self.num_shot = num_shot
		self.num_shot_eval = num_shot_eval

		# Get the total number of samples for each class needed for a batch of Level 1 datasets
		self.num_of_class_per_dataset = num_shot + num_shot_eval

		self.num_classes = num_classes

	# Convert labels to one-hot format for ease of classification.
	def to_onehot(data, num_classes):
		new_data = np.zeros((data.shape[0], num_classes))
		new_data[np.arange(0, data.shape[0]), data] = 1.
		return new_data
	
	def preprocess_data(self, package):

		# The hierarchial structure of data is quite complicated
		# because we are training a model for each step of another model.

		# Dataset Levels
		# Top Level/Level 0:
		# - D_meta_train
		# - D_meta_test

		# Level 1:
		# - D_train
		# - D_test

		# Level 2:
		# - X_train
		# - X_test
		# - y_train
		# - y_test


		# Since Keras gave me X_train, y_train, X_test, and y_test,
		# I chose to process by (X, y) instead of everything at once
		(X, y) = package

		# For the meta-learner to have equal exposure to each class,
		# each batch of Level 2 data requires an equal sample of each class,
		# therefore, here I'm finding the class with the least samples.
		min_objects = min(Counter(y.argmax(axis = -1).tolist()).values())

		# Randomize order of samples,
		# so repeated epochs that didn't use all the data won't train with the same samples
		perm = np.random.permutation(y.shape[0])
		X = X[perm]
		y = y[perm]

		# Create slices of data based on which class,
		# this is to have easier access to samples from each class
		X_slices = dict((i, X[y.argmax(axis = -1) == i, ...]) for i in range(self.num_classes))
		y_slices = dict((i, y[y.argmax(axis = -1) == i, ...]) for i in range(self.num_classes))

		# Init empty Level 0 dataset
		datasets = []

		# Looping through all samples
		# by steps of needed sample number per Level 1 dataset
		for i in range(0, min_objects, self.num_of_class_per_dataset):

			# Init all Level 2 datasets in each batch of Level 1 datasets
			D_X_train = []
			D_X_test = []
			D_y_train = []
			D_y_test = []

			# Cap loop at minimum object class to ensure even label class distribution
			# and same size Level 1 datasets
			if i + self.num_of_class_per_dataset >= min_objects:
				break

			# Loop through each class
			for j in range(self.num_classes):

				# Append Level 2 train and test samples to the corresponding list
				D_X_train.append(X_slices[j][i:i+self.num_shot, ...])
				D_X_test.append(X_slices[j][i+self.num_shot:i+self.num_of_class_per_dataset, ...])

				D_y_train.append(y_slices[j][i:i+self.num_shot, ...])
				D_y_test.append(y_slices[j][i+self.num_shot:i+self.num_of_class_per_dataset, ...])

			# Structure and Append Level 1 and 2 datasets to the Level 0 list
			datasets.append({
				"train": {
				"X": np.float32(np.concatenate(D_X_train, axis = 0)),
				"y": np.concatenate(D_y_train, axis = 0)
				},
				"test": {
				"X": np.float32(np.concatenate(D_X_test, axis = 0)),
				"y": np.concatenate(D_y_test, axis = 0)
				}})
		return datasets

	def __repr__(self):
		return f"Train Num Shot: {self.num_shot}\nEval Num Shot: {self.num_shot_eval}\nNum Classes: {self.num_classes}"





