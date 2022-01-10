import tensorflow as tf
from helper_functions import *
from structures.learner import Learner
from structures.metalearner import *
from data import DataLoader

class TrainSystem():

	def __init__(self, args):
		self.losses = []
		self.accuracies = []

		self.BATCH_SIZE = args["batch_size"]
		self.LEARNER_EPOCHS = args["learner_epochs"]
		self.NUM_CLASSES = args["num_classes"],
		self.LEARNER_STEPS_PER_EPOCH = (args["num_shot"] * args["num_classes"]) // args["batch_size"]
		self.STEPS_PER_EPOCH = args["steps_per_epoch_limit"]
		self.USE_BIAS = args["use_bias"]
		self.EPOCHS = args["epochs"]
		self.args = args
		self.learner_params = args["learner_params"]

		self.crossentropy = tf.keras.losses.CategoricalCrossentropy()
		self.meta_optimizer = tf.keras.optimizers.Adam()
		self.meta_learner = MetaLearner(args, self.learner_params)
		self.learner = Learner(args)
		self.data_loader = DataLoader(args)

		self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.get_fashion_mnist()

	# Function for training a learner with a meta-learner optimizer
	def train_learner(self, D_train):
		"""
		INPUTS:
		D_train (a Level 1 dataset containing X and y samples)
		- train_X
		- train_y
		metalearner (currently training metalearner)
		learner (new learner model to be trained in this function)

		OUTPUTS:
		cI (latest learenr weights optimized by metalearner, which is to be used to get the loss for the metalearner step)
		"""

		# Initialize lists and newest cI from the learner model
		prev_hidden_states = [None]

		cI = tf.concat([tf.reshape(w, (-1,)) for w in self.learner.model.trainable_weights], axis = 0)

		prev_cI = [cI]

		# Loop through each step
		for epoch in range(self.LEARNER_EPOCHS):

			for step in range(self.LEARNER_STEPS_PER_EPOCH):

				# Get current batch of X and y
				train_X = D_train["X"][step * self.BATCH_SIZE: (step + 1) * self.BATCH_SIZE]
				train_y = D_train["y"][step * self.BATCH_SIZE: (step + 1) * self.BATCH_SIZE]

				# Since the gradients are individually calculated each loop,
				# I can reshape the cI BEFORE the gradient tape
				# and then flatten the resulting gradient (AFTER the gradient tape),
				# which then returns a flattened cI.

				# Therefore,
				# 1. create a separate function that reshapes the cI
				# 2. remove the separation part from functional_learner()
				# 3. implement process into code

				# Update: Did solve problem of gradient going to NaN,
				# but now gradient is just going to 0 and the model weights aren't budging.

				# Reshape latest learner weights 
				splitted_cI = reshape_weight(prev_cI[-1], self.learner.model)

				# Create gradient tape to watch learner weights
				with tf.GradientTape() as gt:
					gt.watch(splitted_cI)

					# Call functional learner and calculate loss
					pred = self.learner.functional_learner(train_X, splitted_cI)
					loss = self.crossentropy(train_y, pred)

				# Calculate and flatten gradients for the latest weights
				gradients = gt.gradient(loss, splitted_cI)
				flattened_gradients = tf.concat([tf.reshape(g, (-1,)) for g in gradients], axis = 0)

				# Preprocess gradients and loss for metalearner
				grad_prep = preprocess_grad_or_loss(tf.cast(flattened_gradients, dtype = tf.float32))
				loss_prep = preprocess_grad_or_loss(tf.reduce_mean(loss)[tf.newaxis,...])

				# If this is first step, pass in cI (initial learner weights) as initial cell state for the Meta LSTM Cell
				if prev_hidden_states[-1] is None:
					cI, hidden_states = self.meta_learner([loss_prep, grad_prep, flattened_gradients], hidden_states = prev_hidden_states[-1], c_i = cI)
				else:
					cI, hidden_states = self.meta_learner([loss_prep, grad_prep, flattened_gradients], hidden_states = prev_hidden_states[-1])

				# Append results in this step
				prev_hidden_states.append(hidden_states)
				prev_cI.append(cI)

		return prev_cI[-1]

	# Function for training a metalearner for one step
	def train_meta_learner(self, D_train, D_test):
		"""
		INPUTS:
		D_train (used to train learner)
		- train_X
		- train_y
		D_test (used to evaluate learner and update metalearner)
		- test_X
		- test_y
		metalearner (metalearner to be trained)

		OUTPUTS:
		loss (current loss)
		acc (current accuracy)
		splitted_cI_meta (latest learner weights)
		"""

		# Get Level 2 X and y samples for testing
		test_X, test_y = D_test["X"], D_test["y"]

		# Initialize new learner weights for training
		self.learner.reset_model()

		# Create gradient tape context to watch metalearner's weights
		with tf.GradientTape() as meta_gt:

			meta_gt.watch(self.meta_learner.trainable_variables)

			# Train learner
			cI = self.train_learner(D_train)

			# Evaluate latest trained learner and get loss and acc
			splitted_cI_meta = reshape_weight(cI, self.learner.model)
			test_pred = self.learner.functional_learner(test_X, splitted_cI_meta)
			loss = self.crossentropy(test_y, test_pred)
			acc = accuracy_try(test_y, test_pred)

		# Calculate and apply gradients for metalearner
		# based on loss from latest learner
		meta_grads = meta_gt.gradient(loss, self.meta_learner.trainable_variables)
		self.meta_optimizer.apply_gradients(zip(meta_grads, self.meta_learner.trainable_variables))

		return loss, acc, splitted_cI_meta

	# Over-arching function for training a metalearner
	def train(self):

		for epoch in range(self.EPOCHS):


			# The reason for creating new D_meta_train is
			# because preprocess_data() randomizes the order of
			# the X and y samples, thereby allowing new samples to be used
			D_meta_train = self.data_loader.preprocess_data((self.X_train, self.y_train))

			# Create new training dataset generator for each epoch
			train_generator = self.data_loader.create_flow(D_meta_train)

			for step, D in enumerate(train_generator):

				# Enforce a manual limit on steps per epoch
				if step >= self.STEPS_PER_EPOCH:
					break

				# Parse Level 1 datasets from a Level 0 sample
				D_train, D_test = D["train"], D["test"]

				# Train meta learner for one epoch and get most recent loss and accuracy
				curr_loss, curr_acc, c = self.train_meta_learner(D_train, D_test)

				# Keep track of loss and accuracy
				self.losses.append(curr_loss)
				self.accuracies.append(curr_acc)

				print(f"EPOCH {epoch}; STEP {step};\nLatest loss: {curr_loss}\nLatest accuracy: {curr_acc}\n")

		return c